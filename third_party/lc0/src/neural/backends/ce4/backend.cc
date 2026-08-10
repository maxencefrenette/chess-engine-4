/*
  This file is part of Leela Chess Zero.

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include <algorithm>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "chess_engine_4/inference.h"
#include "neural/backend.h"
#include "neural/encoder.h"
#include "neural/register.h"
#include "neural/shared_params.h"
#include "utils/exception.h"
#include "utils/fastmath.h"

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

 private:
  struct Entry {
    std::vector<chess_engine_4::inference::InputPlane> planes;
    std::vector<Move> legal_moves;
    EvalResultPtr result;
    int transform;
  };

  Ce4Backend* backend_;
  std::vector<Entry> entries_;
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
    backend_options.CheckAllOptionsRead("ce4");

    model_ = chess_engine_4::inference::Model::Load(
        weights_path_, gpu, maximum_batch_size_);
    UpdateConfiguration(options);
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
  std::unique_ptr<chess_engine_4::inference::Model> model_;
  std::mutex model_mutex_;
  int maximum_batch_size_;
  float policy_temperature_ = 1.0f;
  FillEmptyHistory history_fill_ = FillEmptyHistory::NO;
  const std::string backend_options_;
  const std::string weights_path_;

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
  const int batch_size = static_cast<int>(entries_.size());
  if (batch_size == 0) return;

  const int policy_size = backend_->model_->info().policy_size;
  std::vector<chess_engine_4::inference::InputPlane> planes;
  planes.reserve(batch_size * kInputPlanes);
  for (const Entry& entry : entries_) {
    planes.insert(planes.end(), entry.planes.begin(), entry.planes.end());
  }
  std::vector<float> policy(batch_size * policy_size);
  std::vector<float> wdl(batch_size * 3);
  std::vector<float> moves_left(batch_size);
  {
    std::lock_guard lock(backend_->model_mutex_);
    backend_->model_->Evaluate(
        planes, batch_size,
        {.policy = policy, .wdl = wdl, .moves_left = moves_left});
  }

  for (int sample = 0; sample < batch_size; ++sample) {
    Entry& entry = entries_[sample];
    if (entry.result.q) {
      *entry.result.q = wdl[sample * 3] - wdl[sample * 3 + 2];
    }
    if (entry.result.d) *entry.result.d = wdl[sample * 3 + 1];
    if (entry.result.m) *entry.result.m = moves_left[sample];
    if (entry.result.p.empty()) continue;
    if (entry.result.p.size() != entry.legal_moves.size()) {
      throw Exception("ce4 policy output does not match legal move count");
    }

    const float* logits = policy.data() + sample * policy_size;
    float maximum = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < entry.legal_moves.size(); ++i) {
      const int index = MoveToNNIndex(entry.legal_moves[i], entry.transform);
      entry.result.p[i] = logits[index];
      maximum = std::max(maximum, entry.result.p[i]);
    }
    float total = 0.0f;
    for (float& value : entry.result.p) {
      value = FastExp((value - maximum) * backend_->policy_temperature_);
      total += value;
    }
    const float scale = total > 0.0f ? 1.0f / total : 1.0f;
    for (float& value : entry.result.p) value *= scale;
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
