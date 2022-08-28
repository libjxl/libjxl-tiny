// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_BASE_DATA_PARALLEL_H_
#define ENCODER_BASE_DATA_PARALLEL_H_

// Portable, low-overhead C++11 ThreadPool alternative to OpenMP for
// data-parallel computations.

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <condition_variable>  //NOLINT
#include <mutex>               //NOLINT
#include <thread>              //NOLINT
#include <vector>

#include "encoder/base/bits.h"
#include "encoder/base/status.h"

#if JXL_COMPILER_MSVC
// suppress warnings about the const & applied to function types
#pragma warning(disable : 4180)
#endif

/** Return code used in the JxlParallel* functions as return value. A value
 * of 0 means success and any other value means error. The special value
 * JXL_PARALLEL_RET_RUNNER_ERROR can be used by the runner to indicate any
 * other error.
 */
typedef int JxlParallelRetCode;

/**
 * General error returned by the JxlParallelRunInit function to indicate
 * an error.
 */
#define JXL_PARALLEL_RET_RUNNER_ERROR (-1)

/**
 * Parallel run initialization callback. See JxlParallelRunner for details.
 *
 * This function MUST be called by the JxlParallelRunner only once, on the
 * same thread that called JxlParallelRunner, before any parallel execution.
 * The purpose of this call is to provide the maximum number of threads that the
 * JxlParallelRunner will use, which can be used by JPEG XL to allocate
 * per-thread storage if needed.
 *
 * @param jpegxl_opaque the @p jpegxl_opaque handle provided to
 * JxlParallelRunner() must be passed here.
 * @param num_threads the maximum number of threads. This value must be
 * positive.
 * @return 0 if the initialization process was successful.
 * @return an error code if there was an error, which should be returned by
 * JxlParallelRunner().
 */
typedef JxlParallelRetCode (*JxlParallelRunInit)(void* jpegxl_opaque,
                                                 size_t num_threads);

/**
 * Parallel run data processing callback. See JxlParallelRunner for details.
 *
 * This function MUST be called once for every number in the range [start_range,
 * end_range) (including start_range but not including end_range) passing this
 * number as the @p value. Calls for different value may be executed from
 * different threads in parallel.
 *
 * @param jpegxl_opaque the @p jpegxl_opaque handle provided to
 * JxlParallelRunner() must be passed here.
 * @param value the number in the range [start_range, end_range) of the call.
 * @param thread_id the thread number where this function is being called from.
 * This must be lower than the @p num_threads value passed to
 * JxlParallelRunInit.
 */
typedef void (*JxlParallelRunFunction)(void* jpegxl_opaque, uint32_t value,
                                       size_t thread_id);

namespace jxl {

// Main helper class implementing the ::JxlParallelRunner interface.
class ThreadParallelRunner {
 public:
  // ::JxlParallelRunner interface.
  static JxlParallelRetCode Runner(void* runner_opaque, void* jpegxl_opaque,
                                   JxlParallelRunInit init,
                                   JxlParallelRunFunction func,
                                   uint32_t start_range, uint32_t end_range);

  // Starts the given number of worker threads and blocks until they are ready.
  // "num_worker_threads" defaults to one per hyperthread. If zero, all tasks
  // run on the main thread.
  explicit ThreadParallelRunner(
      int num_worker_threads = std::thread::hardware_concurrency());

  // Waits for all threads to exit.
  ~ThreadParallelRunner();

 private:
  // After construction and between calls to Run, workers are "ready", i.e.
  // waiting on worker_start_cv_. They are "started" by sending a "command"
  // and notifying all worker_start_cv_ waiters. (That is why all workers
  // must be ready/waiting - otherwise, the notification will not reach all of
  // them and the main thread waits in vain for them to report readiness.)
  using WorkerCommand = uint64_t;

  // Special values; all others encode the begin/end parameters. Note that all
  // these are no-op ranges (begin >= end) and therefore never used to encode
  // ranges.
  static constexpr WorkerCommand kWorkerWait = ~1ULL;
  static constexpr WorkerCommand kWorkerOnce = ~2ULL;
  static constexpr WorkerCommand kWorkerExit = ~3ULL;

  // Calls f(task, thread). Used for type erasure of Func arguments. The
  // signature must match JxlParallelRunFunction, hence a void* argument.
  template <class Closure>
  static void CallClosure(void* f, const uint32_t task, const size_t thread) {
    (*reinterpret_cast<const Closure*>(f))(task, thread);
  }

  void WorkersReadyBarrier() {
    std::unique_lock<std::mutex> lock(mutex_);
    // Typically only a single iteration.
    while (workers_ready_ != threads_.size()) {
      workers_ready_cv_.wait(lock);
    }
    workers_ready_ = 0;

    // Safely handle spurious worker wakeups.
    worker_start_command_ = kWorkerWait;
  }

  // Precondition: all workers are ready.
  void StartWorkers(const WorkerCommand worker_command) {
    mutex_.lock();
    worker_start_command_ = worker_command;
    // Workers will need this lock, so release it before they wake up.
    mutex_.unlock();
    worker_start_cv_.notify_all();
  }

  // Attempts to reserve and perform some work from the global range of tasks,
  // which is encoded within "command". Returns after all tasks are reserved.
  static void RunRange(ThreadParallelRunner* self, const WorkerCommand command,
                       const int thread);

  static void ThreadFunc(ThreadParallelRunner* self, int thread);

  // Unmodified after ctor, but cannot be const because we call thread::join().
  std::vector<std::thread> threads_;

  const uint32_t num_worker_threads_;  // == threads_.size()
  const uint32_t num_threads_;

  std::atomic<int> depth_{0};  // detects if Run is re-entered (not supported).

  std::mutex mutex_;  // guards both cv and their variables.
  std::condition_variable workers_ready_cv_;
  uint32_t workers_ready_ = 0;
  std::condition_variable worker_start_cv_;
  WorkerCommand worker_start_command_;

  // Written by main thread, read by workers (after mutex lock/unlock).
  JxlParallelRunFunction data_func_;
  void* jpegxl_opaque_;

  // Updated by workers; padding avoids false sharing.
  uint8_t padding1[64];
  std::atomic<uint32_t> num_reserved_{0};
  uint8_t padding2[64];
};

class ThreadPool {
 public:
  // Starts the given number of worker threads and blocks until they are ready.
  // "num_worker_threads" defaults to one per hyperthread. If zero, all tasks
  // run on the main thread.
  explicit ThreadPool(
      int num_worker_threads = std::thread::hardware_concurrency())
      : runner_(num_worker_threads) {}

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator&(const ThreadPool&) = delete;

  // Runs init_func(num_threads) followed by data_func(task, thread) on worker
  // thread(s) for every task in [begin, end). init_func() must return a Status
  // indicating whether the initialization succeeded.
  // "thread" is an integer smaller than num_threads.
  // Not thread-safe - no two calls to Run may overlap.
  // Subsequent calls will reuse the same threads.
  //
  // Precondition: begin <= end.
  template <class InitFunc, class DataFunc>
  Status Run(uint32_t begin, uint32_t end, const InitFunc& init_func,
             const DataFunc& data_func, const char* caller = "") {
    JXL_ASSERT(begin <= end);
    if (begin == end) return true;
    RunCallState<InitFunc, DataFunc> call_state(init_func, data_func);
    // The runner_ uses the C convention and returns 0 in case of error, so we
    // convert it to a Status.
    return ThreadParallelRunner::Runner(
               &runner_, static_cast<void*>(&call_state),
               &call_state.CallInitFunc, &call_state.CallDataFunc, begin,
               end) == 0;
  }

  // Use this as init_func when no initialization is needed.
  static Status NoInit(size_t num_threads) { return true; }

 private:
  // class holding the state of a Run() call to pass to the runner_ as an
  // opaque_jpegxl pointer.
  template <class InitFunc, class DataFunc>
  class RunCallState final {
   public:
    RunCallState(const InitFunc& init_func, const DataFunc& data_func)
        : init_func_(init_func), data_func_(data_func) {}

    // JxlParallelRunInit interface.
    static int CallInitFunc(void* jpegxl_opaque, size_t num_threads) {
      const auto* self =
          static_cast<RunCallState<InitFunc, DataFunc>*>(jpegxl_opaque);
      // Returns -1 when the internal init function returns false Status to
      // indicate an error.
      return self->init_func_(num_threads) ? 0 : -1;
    }

    // JxlParallelRunFunction interface.
    static void CallDataFunc(void* jpegxl_opaque, uint32_t value,
                             size_t thread_id) {
      const auto* self =
          static_cast<RunCallState<InitFunc, DataFunc>*>(jpegxl_opaque);
      return self->data_func_(value, thread_id);
    }

   private:
    const InitFunc& init_func_;
    const DataFunc& data_func_;
  };

  ThreadParallelRunner runner_;
};

template <class InitFunc, class DataFunc>
Status RunOnPool(ThreadPool* pool, const uint32_t begin, const uint32_t end,
                 const InitFunc& init_func, const DataFunc& data_func,
                 const char* caller) {
  if (pool == nullptr) {
    ThreadPool default_pool(0);
    return default_pool.Run(begin, end, init_func, data_func, caller);
  } else {
    return pool->Run(begin, end, init_func, data_func, caller);
  }
}

}  // namespace jxl
#if JXL_COMPILER_MSVC
#pragma warning(default : 4180)
#endif

#endif  // ENCODER_BASE_DATA_PARALLEL_H_
