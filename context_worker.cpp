#include <llama.h>
#include <queue>
#include <mutex>
#include "xxhash.h"
#include <thread>
#define MAX_RESPONSE_LENGTH 4096


typedef struct LlamaModelCtx {
    llama_model *model;
    llama_context *ctx;
} llama_model_ctx_t;

class ContextWorker {
public:
    ContextWorker(int id, llama_model_ctx_t* context): context_id(id), 
    ctx(context->ctx), stop(false), preempt(false), in_use(false),
    model_mutex(PTHREAD_MUTEX_INITIALIZER), model(context->model) {
        worker = std::thread([this]{this->run();});
    }
    ~ContextWorker() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        cv.notify_one();
        worker.join();
    }
    void enqueue(std::string prompt) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        tasks.push(prompt);
        cv.notify_one(); 
    }
private:
    int context_id;
    struct llama_context *ctx;
    std::queue<std::string> tasks;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::thread worker;
    bool stop;
    bool preempt;
    bool in_use;
    int32_t current_priority;
    pthread_mutex_t model_mutex;
    std::unordered_map<std::string, std::vector<uint8_t>> cache_state_map;


    struct llama_model * model;

    void run() {
        while(true) {
            std::string prompt;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                cv.wait(lock, [this]{ return stop || !tasks.empty();});
                if(stop && tasks.empty()) return;
                prompt = std::move(tasks.front());
                tasks.pop();
                int32_t priority = get_priority(prompt.c_str());
                if(priority > current_priority && in_use) {
                    tasks.push(std::move(prompt));
                    continue;
                }
                current_priority = priority;
            }
            in_use = true;
            int seed = 10;
            uint64_t prompt_hash = XXH64(prompt.c_str(), prompt.size(), seed);
            char cache_key[32];
            snprintf(cache_key, sizeof(cache_key), "%016llx", prompt_hash);
            printf("Hashed prompt: %s\n", cache_key);
            auto it = cache_state_map.find(cache_key);
            bool update_cache = true;
            if (it != cache_state_map.end()) {
                printf("[%d]Cache Found for %s\n", context_id, cache_key);
                const std::vector<uint8_t>& saved_state = it->second;
                llama_state_set_data(ctx, saved_state.data(), saved_state.size());
                update_cache = false;
            }
            printf("[%d]Cache size: %zu\n", context_id, cache_state_map.size());
            auto start_time = std::chrono::high_resolution_clock::now();
            ai_run_inference(ctx, prompt.c_str(), cache_key, update_cache);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            // printf("TIME TAKEN FOR INFERENCE WITH HASH %s IS : %lld\n", prompt_hash, t_ms);
            printf("[%d]TIME TAKEN FOR INFERENCE IS : %lld\n", context_id, t_ms);
            llama_kv_self_clear(ctx);
        }
    }


    // Thread-safe vocab access
    const llama_vocab* get_vocab_safe() {
        pthread_mutex_lock(&model_mutex);
        const llama_vocab* vocab = llama_model_get_vocab(model);
        pthread_mutex_unlock(&model_mutex);
        return vocab;
    }

    int32_t get_priority(const char* prompt) {
        const llama_vocab* vocab = get_vocab_safe();
        const int n_prompt = -llama_tokenize(vocab, prompt, strlen(prompt), NULL, 0, true, true);
        return n_prompt;
    }


    void ai_run_inference(struct llama_context* ctx, 
                            const char* prompt, 
                            std:: string prompt_hash,
                            bool update_cache) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        int n_predict = 20;
        const llama_vocab* vocab = get_vocab_safe();
        const int n_prompt = -llama_tokenize(vocab, prompt, strlen(prompt), NULL, 0, true, true);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        printf("[%d]TIME TAKEN FOR Tokenize count IS : %lld\n", context_id, t_ms);
        
        
        std::vector<llama_token> prompt_tokens(n_prompt);
        if (llama_tokenize(vocab, prompt, strlen(prompt), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
            fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
            return;
        }

        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = false;
        llama_sampler * smpl = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        

        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        llama_token new_token_id;
        
        // Use the per-context buffer instead of malloc
        char response[MAX_RESPONSE_LENGTH];
        memset(response, 0, MAX_RESPONSE_LENGTH);
        size_t response_len = 0;

        for (int n_pos = batch.n_tokens; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
            start_time = std::chrono::high_resolution_clock::now();
            if (llama_decode(ctx, batch)) {
                fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
                llama_sampler_free(smpl);
                return;
            }


            end_time = std::chrono::high_resolution_clock::now();
            t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            // printf("TIME TAKEN FOR INFERENCE WITH HASH %s IS : %lld\n", prompt_hash, t_ms);
            printf("[%d]TIME TAKEN FOR Decode IS : %lld\n", context_id, t_ms);
            
            n_pos += batch.n_tokens;
            
            start_time = std::chrono::high_resolution_clock::now();
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
                    return;
                }
                
                // Safe string concatenation
                if (response_len + n < MAX_RESPONSE_LENGTH - 1) {
                    memcpy(response + response_len, buf, n);
                    response_len += n;
                    response[response_len] = '\0';
                }

                batch = llama_batch_get_one(&new_token_id, 1);
            }

            end_time = std::chrono::high_resolution_clock::now();
            t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            // printf("TIME TAKEN FOR INFERENCE WITH HASH %s IS : %lld\n", prompt_hash, t_ms);
            printf("[%d]TIME TAKEN FOR Last IS : %lld\n", context_id, t_ms);
        }
        
        llama_sampler_free(smpl);
        printf("Response: %s\n", response);
        
        if(update_cache) {
            printf("Updating Cache\n");
            size_t state_size = llama_state_get_size(ctx);
            printf("Size of state cache is: %zu\n", state_size);
            std::vector<uint8_t> state_buf(state_size);
            llama_state_get_data(ctx, state_buf.data(), state_size);
            cache_state_map[prompt_hash] = std::move(state_buf);
            printf("Cache Saved\n");
        }
        in_use = false;
    }



};