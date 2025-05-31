#include <chrono>
#include <llama.h>
#include "threadPool.cpp"
#include "llama_schedular.cpp"
#include <pthread.h>
#include <atomic>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <vector>
#include <unistd.h>
#include <future>

#define LLAMA_POOL_SIZE 4
#define MAX_RESPONSE_LENGTH 4096
typedef int32_t llama_tokens;


typedef struct {
    struct llama_context *ctx;
    std::atomic<bool> in_use;
    uint32_t index;
    char response_buffer[MAX_RESPONSE_LENGTH];
} llama_context_t;

std::vector<struct llama_context*> contexts;

static llama_context_t llama_pool[LLAMA_POOL_SIZE];
static pthread_mutex_t llama_pool_mutex = PTHREAD_MUTEX_INITIALIZER;

static pthread_mutex_t model_mutex = PTHREAD_MUTEX_INITIALIZER;

static struct llama_model * model = nullptr;

typedef struct {
    struct llama_context* ctx;
    int pool_index;
} llama_ctx_handle_t;

static llama_ctx_handle_t acquire_llama_ctx() {
    llama_ctx_handle_t handle = {NULL, -1};
    
    pthread_mutex_lock(&llama_pool_mutex);
    for (int i = 0; i < LLAMA_POOL_SIZE; i++) {
        bool expected = false;
        if (llama_pool[i].in_use.compare_exchange_strong(expected, true)) {
            printf("Acquired LLaMA context %d\n", i);
            handle.ctx = llama_pool[i].ctx;
            handle.pool_index = i;
            pthread_mutex_unlock(&llama_pool_mutex);
            // pthread_mutex_lock(&llama_pool[i].ctx_mutex);
            return handle;
        }
    }
    pthread_mutex_unlock(&llama_pool_mutex);
    return handle;
}

static void release_llama_ctx(llama_ctx_handle_t handle) {
    if (handle.pool_index < 0 || handle.pool_index >= LLAMA_POOL_SIZE) {
        return;
    }
    
    printf("Releasing LLaMA context %d\n", handle.pool_index);
    
    // First unlock the context mutex
    // pthread_mutex_unlock(&llama_pool[handle.pool_index].ctx_mutex);
    
    // Then mark as available
    llama_pool[handle.pool_index].in_use.store(false);
}

// Thread-safe vocab access
static const llama_vocab* get_vocab_safe() {
    pthread_mutex_lock(&model_mutex);
    const llama_vocab* vocab = llama_model_get_vocab(model);
    pthread_mutex_unlock(&model_mutex);
    return vocab;
}



char * ai_run_inference(struct llama_context* ctx, const char* prompt) {
    int n_predict = 20;
    
    const llama_vocab* vocab = get_vocab_safe();
    const int n_prompt = -llama_tokenize(vocab, prompt, strlen(prompt), NULL, 0, true, true);
    
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt, strlen(prompt), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return NULL;
    }

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    llama_token new_token_id;
    
    // Clear KV cache to ensure clean state
    // llama_kv_cache_clear(handle.ctx);
    // Use the per-context buffer instead of malloc
    char response[MAX_RESPONSE_LENGTH];
    memset(response, 0, MAX_RESPONSE_LENGTH);
    size_t response_len = 0;

    for (int n_pos = batch.n_tokens; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            llama_sampler_free(smpl);
            return NULL;
        }

        n_pos += batch.n_tokens;

        {
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                llama_sampler_free(smpl);
                return NULL;
            }
            
            // Safe string concatenation
            if (response_len + n < MAX_RESPONSE_LENGTH - 1) {
                memcpy(response + response_len, buf, n);
                response_len += n;
                response[response_len] = '\0';
            }

            batch = llama_batch_get_one(&new_token_id, 1);
        }
    }
    
    llama_sampler_free(smpl);
    printf("Response: %s\n", response);
    
    // Allocate new memory for return value
    char* result = (char*)malloc(strlen(response) + 1);
    strcpy(result, response);
    
    return result;
}


int main() {

    const char *llama_model_path = "./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf";

    // Initialize LLaMA
    llama_backend_init();
    auto lmparams = llama_model_default_params();
    lmparams.n_gpu_layers = 0;
    
    model = llama_model_load_from_file(llama_model_path, lmparams);

    // pthread_mutex_init(&llama_pool_mutex, NULL);
    llama_context_params lcparams = llama_context_default_params();
    lcparams.n_ctx      = 4096;
    lcparams.n_threads  = 1;  // Reduced from 8 for better concurrency
    lcparams.flash_attn = false;
    lcparams.no_perf = false;

    // pthread_mutex_lock(&llama_pool_mutex);
    // for(int i = 0; i < LLAMA_POOL_SIZE; i++) {
        // llama_pool[i].ctx = llama_init_from_model(model, lcparams);
        // llama_pool[i].in_use = false;
        // llama_pool[i].index = i;
    // }
    // pthread_mutex_unlock(&llama_pool_mutex);
    for(int i = 0; i < LLAMA_POOL_SIZE; i++) {
        struct llama_context *ctx = llama_init_from_model(model, lcparams);
        contexts.push_back(ctx);
    }

    printf("LLaMa Model initialized checking threadpool now\n");
    std::vector<std::string> messages = {"Hello", "How are you?", "what is india?", "how is Laos country?"};
    LlamaSchedular scheduler(contexts);

    auto start_time = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < LLAMA_POOL_SIZE; i++) {
        std::string message = messages[i%4];
        printf("on Index: %d Using message: %s\n", i, message.c_str());
        scheduler.schedule([message, start_time, i] (struct llama_context *ctx) {
            auto task_start = std::chrono::high_resolution_clock::now();
            auto task_start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                task_start - start_time).count();
            auto result = ai_run_inference(ctx, message.c_str());
            printf("Result for inference: %s\n", result);
            auto task_end = std::chrono::high_resolution_clock::now();
            auto task_end_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                task_end - start_time).count();
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                task_end - task_start).count();
            
            printf("Task %d completed at: %ld ms (took %ld ms)\n", 
                   i, task_end_ms, duration_ms);
        });
    }
    
    
}