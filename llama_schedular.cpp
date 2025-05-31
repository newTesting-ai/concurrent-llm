#include <llama.h>
#include <vector>
#include "context_worker.cpp"

class LlamaSchedular {
    public:
        LlamaSchedular(std::vector<struct llama_context*> contexts): index(0) {
            for(size_t i = 0; i < contexts.size(); i++) {
                workers.emplace_back(new ContextWorker(i, contexts[i]));
            }
        }
        ~LlamaSchedular() {
            for(auto &w: workers) {
                delete w;
            }
        }

        void schedule(std::function<void(struct llama_context*)> job) {
            std::unique_lock<std::mutex> lock(index_mutex);
            workers[index]->enqueue(job);
            index = (index + 1) % workers.size();
        }
    private:
        int index;
        std::vector<ContextWorker *> workers;
        std::mutex index_mutex;
};