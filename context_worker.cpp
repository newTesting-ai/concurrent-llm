#include <llama.h>
#include <queue>
#include <mutex>
#include <thread>

class ContextWorker {
public:
    ContextWorker(int id, struct llama_context* ctx): context_id(id), ctx(ctx), stop(false) {
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
    void enqueue(std::function<void(struct llama_context*)> job) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        tasks.push(job);
        cv.notify_one(); 
    }
private:
    int context_id;
    struct llama_context *ctx;
    std::queue<std::function<void(struct llama_context*)>> tasks;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::thread worker;
    bool stop;

    void run() {
        while(true) {
            std::function<void(struct llama_context*)> job;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                cv.wait(lock, [this]{ return stop || !tasks.empty();});
                if(stop && tasks.empty()) return;
                job = std::move(tasks.front());
                tasks.pop();
            }
            job(ctx);
        }
    }

};