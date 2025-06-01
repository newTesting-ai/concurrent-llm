#include <llama.h>
#include <vector>
#include "context_worker.cpp"


class LlamaSchedular {
    public:
        LlamaSchedular(std::vector<std::unique_ptr<llama_model_ctx_t>> &contexts): index(0) {
            for(size_t i = 0; i < contexts.size(); i++) {
                workers.emplace_back(std::make_unique<ContextWorker>(i, contexts[i].get()));
            }
        }
        ~LlamaSchedular() {
            workers.clear();
        }

        void schedule(std::string prompt) {
            std::unique_lock<std::mutex> lock(index_mutex);
            workers[index]->enqueue(prompt);
            index = (index + 1) % workers.size();
        }
    private:
        int index;
        std::vector<std::unique_ptr<ContextWorker>> workers;
        std::mutex index_mutex;
};