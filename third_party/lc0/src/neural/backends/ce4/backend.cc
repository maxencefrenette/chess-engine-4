/*
  This file is part of Leela Chess Zero.

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "chess_engine_4/inference.h"
#include "neural/backend.h"
#include "neural/encoder.h"
#include "neural/register.h"
#include "neural/shared_params.h"
#include "utils/exception.h"
#include "utils/fastmath.h"
#include "utils/logging.h"

namespace lczero {
namespace {

FillEmptyHistory ParseHistoryFill(const std::string& history_fill) {
  if (history_fill == "fen_only") return FillEmptyHistory::FEN_ONLY;
  if (history_fill == "always") return FillEmptyHistory::ALWAYS;
  if (history_fill == "no") return FillEmptyHistory::NO;
  throw Exception("Unsupported history fill mode: " + history_fill);
}

class Ce4Backend;

class Ce4Computation final : public BackendComputation {
 public:
  explicit Ce4Computation(Ce4Backend* backend) : backend_(backend) {}

  size_t UsedBatchSize() const override { return entries_.size(); }
  AddInputResult AddInput(const EvalPosition& position,
                          EvalResultPtr result) override;
  void ComputeBlocking() override;

  void NotifyReady(std::exception_ptr error = nullptr) {
    {
      std::lock_guard lock(mutex_);
      error_ = error;
      ready_ = true;
    }
    ready_cv_.notify_one();
  }

  void Wait() {
    std::unique_lock lock(mutex_);
    ready_cv_.wait(lock, [this] { return ready_; });
    if (error_) std::rethrow_exception(error_);
  }

 private:
  struct Entry {
    std::vector<chess_engine_4::inference::InputPlane> planes;
    std::vector<Move> legal_moves;
    EvalResultPtr result;
    int transform;
  };

  Ce4Backend* backend_;
  std::vector<Entry> entries_;
  std::mutex mutex_;
  std::condition_variable ready_cv_;
  std::exception_ptr error_;
  bool ready_ = false;

  friend class Ce4Backend;
};

class Ce4Backend final : public Backend {
 public:
  explicit Ce4Backend(const OptionsDict& options)
      : backend_options_(
            options.Get<std::string>(SharedBackendParams::kBackendOptionsId)),
        weights_path_(
            options.Get<std::string>(SharedBackendParams::kWeightsId)) {
    OptionsDict backend_options;
    backend_options.AddSubdictFromString(backend_options_);
    const int gpu = backend_options.GetOrDefault<int>("gpu", 0);
    maximum_batch_size_ =
        backend_options.GetOrDefault<int>("max_batch", 256);
    if (maximum_batch_size_ <= 0 || maximum_batch_size_ > 1024) {
      throw Exception("ce4 max_batch must be between 1 and 1024");
    }
    batch_wait_us_ = backend_options.GetOrDefault<int>("batch_wait_us", 200);
    if (batch_wait_us_ < 0 || batch_wait_us_ > 100000) {
      throw Exception("ce4 batch_wait_us must be between 0 and 100000");
    }
    backend_options.CheckAllOptionsRead("ce4");

    model_ = chess_engine_4::inference::Model::Load(
        weights_path_, gpu, maximum_batch_size_);
    UpdateConfiguration(options);
    worker_ = std::thread([this] { Worker(); });
  }

  ~Ce4Backend() override {
    {
      std::lock_guard lock(queue_mutex_);
      abort_ = true;
    }
    queue_cv_.notify_one();
    if (worker_.joinable()) worker_.join();
    CERR << "ce4_batch_stats weights=" << weights_path_
         << " calls=" << inference_calls_
         << " positions=" << evaluated_positions_
         << " average_batch="
         << (inference_calls_ == 0
                 ? 0.0
                 : static_cast<double>(evaluated_positions_) / inference_calls_)
         << " max_batch=" << largest_batch_;
  }

  BackendAttributes GetAttributes() const override {
    return BackendAttributes{
        .has_mlh = true,
        .has_wdl = true,
        .runs_on_cpu = false,
        .suggested_num_search_threads = 1,
        .recommended_batch_size = std::min(maximum_batch_size_, 256),
        .maximum_batch_size = maximum_batch_size_,
    };
  }

  std::unique_ptr<BackendComputation> CreateComputation() override {
    return std::make_unique<Ce4Computation>(this);
  }

  UpdateConfigurationResult UpdateConfiguration(
      const OptionsDict& options) override {
    Backend::UpdateConfiguration(options);
    if (backend_options_ !=
            options.Get<std::string>(SharedBackendParams::kBackendOptionsId) ||
        weights_path_ !=
            options.Get<std::string>(SharedBackendParams::kWeightsId)) {
      return NEED_RESTART;
    }
    policy_temperature_ =
        1.0f / options.Get<float>(SharedBackendParams::kPolicySoftmaxTemp);
    history_fill_ = ParseHistoryFill(
        options.Get<std::string>(SharedBackendParams::kHistoryFill));
    return UPDATE_OK;
  }

 private:
  void Enqueue(Ce4Computation* computation) {
    {
      std::lock_guard lock(queue_mutex_);
      queue_.push_back(computation);
      queued_positions_ += computation->entries_.size();
    }
    queue_cv_.notify_one();
    computation->Wait();
  }

  void Worker();
  void Evaluate(const std::vector<Ce4Computation*>& computations);

  std::unique_ptr<chess_engine_4::inference::Model> model_;
  int maximum_batch_size_;
  int batch_wait_us_;
  float policy_temperature_ = 1.0f;
  FillEmptyHistory history_fill_ = FillEmptyHistory::NO;
  const std::string backend_options_;
  const std::string weights_path_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::deque<Ce4Computation*> queue_;
  size_t queued_positions_ = 0;
  bool abort_ = false;
  std::thread worker_;
  size_t inference_calls_ = 0;
  size_t evaluated_positions_ = 0;
  size_t largest_batch_ = 0;

  friend class Ce4Computation;
};

BackendComputation::AddInputResult Ce4Computation::AddInput(
    const EvalPosition& position, EvalResultPtr result) {
  if (entries_.size() >=
      static_cast<size_t>(backend_->maximum_batch_size_)) {
    throw Exception("ce4 computation exceeds configured max_batch");
  }

  int transform = 0;
  InputPlanes encoded = EncodePositionForNN(
      pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE, position.pos, 8,
      backend_->history_fill_, &transform);
  std::vector<chess_engine_4::inference::InputPlane> planes;
  planes.reserve(encoded.size());
  for (const InputPlane& plane : encoded) {
    planes.push_back({plane.mask, plane.value});
  }
  entries_.push_back({
      .planes = std::move(planes),
      .legal_moves = {position.legal_moves.begin(), position.legal_moves.end()},
      .result = result,
      .transform = transform,
  });
  return ENQUEUED_FOR_EVAL;
}

void Ce4Computation::ComputeBlocking() {
  if (entries_.empty()) return;
  backend_->Enqueue(this);
}

void Ce4Backend::Worker() {
  while (true) {
    std::vector<Ce4Computation*> computations;
    {
      std::unique_lock lock(queue_mutex_);
      queue_cv_.wait(lock, [this] { return abort_ || !queue_.empty(); });
      if (abort_ && queue_.empty()) return;
      if (!abort_ && queued_positions_ < static_cast<size_t>(maximum_batch_size_) &&
          batch_wait_us_ > 0) {
        queue_cv_.wait_for(
            lock, std::chrono::microseconds(batch_wait_us_),
            [this] {
              return abort_ ||
                     queued_positions_ >= static_cast<size_t>(maximum_batch_size_);
            });
      }
      size_t batch_size = 0;
      while (!queue_.empty()) {
        const size_t computation_size = queue_.front()->entries_.size();
        if (batch_size != 0 &&
            batch_size + computation_size >
                static_cast<size_t>(maximum_batch_size_)) {
          break;
        }
        computations.push_back(queue_.front());
        queue_.pop_front();
        queued_positions_ -= computation_size;
        batch_size += computation_size;
      }
    }

    std::exception_ptr error;
    try {
      Evaluate(computations);
    } catch (...) {
      error = std::current_exception();
    }
    for (Ce4Computation* computation : computations) {
      computation->NotifyReady(error);
    }
  }
}

void Ce4Backend::Evaluate(
    const std::vector<Ce4Computation*>& computations) {
  int batch_size = 0;
  for (const Ce4Computation* computation : computations) {
    batch_size += static_cast<int>(computation->entries_.size());
  }
  if (batch_size == 0) return;

  const int policy_size = model_->info().policy_size;
  std::vector<chess_engine_4::inference::InputPlane> planes;
  planes.reserve(batch_size * kInputPlanes);
  for (const Ce4Computation* computation : computations) {
    for (const Ce4Computation::Entry& entry : computation->entries_) {
      planes.insert(planes.end(), entry.planes.begin(), entry.planes.end());
    }
  }
  std::vector<float> policy(batch_size * policy_size);
  std::vector<float> wdl(batch_size * 3);
  std::vector<float> moves_left(batch_size);
  model_->Evaluate(planes, batch_size,
                   {.policy = policy, .wdl = wdl, .moves_left = moves_left});
  ++inference_calls_;
  evaluated_positions_ += batch_size;
  largest_batch_ = std::max(largest_batch_, static_cast<size_t>(batch_size));

  int sample = 0;
  for (Ce4Computation* computation : computations) {
    for (Ce4Computation::Entry& entry : computation->entries_) {
      const int sample_index = sample++;
      if (entry.result.q) {
        *entry.result.q =
            wdl[sample_index * 3] - wdl[sample_index * 3 + 2];
      }
      if (entry.result.d) *entry.result.d = wdl[sample_index * 3 + 1];
      if (entry.result.m) *entry.result.m = moves_left[sample_index];
      if (entry.result.p.empty()) continue;
      if (entry.result.p.size() != entry.legal_moves.size()) {
        throw Exception("ce4 policy output does not match legal move count");
      }

      const float* logits = policy.data() + sample_index * policy_size;
      float maximum = -std::numeric_limits<float>::infinity();
      for (size_t i = 0; i < entry.legal_moves.size(); ++i) {
        const int index = MoveToNNIndex(entry.legal_moves[i], entry.transform);
        entry.result.p[i] = logits[index];
        maximum = std::max(maximum, entry.result.p[i]);
      }
      float total = 0.0f;
      for (float& value : entry.result.p) {
        value = FastExp((value - maximum) * policy_temperature_);
        total += value;
      }
      const float scale = total > 0.0f ? 1.0f / total : 1.0f;
      for (float& value : entry.result.p) value *= scale;
    }
  }
}

class Ce4BackendFactory final : public BackendFactory {
 public:
  int GetPriority() const override { return 120; }
  std::string_view GetName() const override { return "ce4"; }
  std::unique_ptr<Backend> Create(const OptionsDict& options) override {
    return std::make_unique<Ce4Backend>(options);
  }
};

[[maybe_unused]] BackendManager::Register ce4_registration(
    std::make_unique<Ce4BackendFactory>());

}  // namespace
}  // namespace lczero
