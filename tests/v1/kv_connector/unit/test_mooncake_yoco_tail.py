# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Producer/consumer token boundaries for YOCO's final-block replay."""

from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnectorScheduler,
    MooncakeConnectorWorker,
    MooncakeXferResponse,
    MooncakeXferResponseStatus,
    PullReqMeta,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request


def scheduler(role, yoco=True):
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16, kv_sharing_fast_prefill=yoco),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="yoco")),
        kv_transfer_config=SimpleNamespace(kv_role=role),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
    )
    return MooncakeConnectorScheduler(
        config, "test", SimpleNamespace(kv_cache_groups=[])
    )


def request(n, role):
    return Request(
        request_id=role,
        prompt_token_ids=list(range(n)),
        sampling_params=SamplingParams(
            max_tokens=8,
            extra_args={
                "kv_transfer_params": {
                    "do_remote_decode": role == "p",
                    "do_remote_prefill": role == "d",
                }
            },
        ),
        pooling_params=None,
    )


@pytest.mark.parametrize(
    "n,prefix",
    [
        (1, 0),
        (15, 0),
        (16, 0),
        (17, 16),
        (32, 16),
        (33, 32),
        (512, 496),
        (513, 512),
        (2048, 2032),
        (2049, 2048),
    ],
)
def test_matching_tail_boundary_and_resume(n, prefix):
    p, d = scheduler("kv_producer"), scheduler("kv_consumer")
    producer, consumer = request(n, "p"), request(n, "d")
    p.on_new_request(producer)
    assert d.get_num_new_matched_tokens(consumer, 0) == (prefix, prefix > 0)
    assert producer.num_prompt_tokens == (prefix or n)
    assert producer.prompt_token_ids == list(range(prefix or n))
    assert list(producer.all_token_ids) == list(range(prefix or n))
    # Repeated admission after preemption must never remove another block.
    p.on_new_request(producer)
    assert producer.num_prompt_tokens == (prefix or n)
    assert d.get_num_new_matched_tokens(consumer, prefix) == (0, False)
    if prefix:
        assert producer.skip_reading_prefix_cache
        assert producer.max_tokens == 1
        assert producer.kv_transfer_params["_p_side_truncated"]
    else:
        assert not producer.skip_reading_prefix_cache
        assert not producer.kv_transfer_params.get("_p_side_truncated")
    # The consumer's public prompt and accounting remain intact.
    assert consumer.num_prompt_tokens == n
    assert consumer.prompt_token_ids == list(range(n))


def test_without_yoco_tail_policy():
    p, d = scheduler("kv_producer", False), scheduler("kv_consumer", False)
    producer, consumer = request(2048, "p"), request(2048, "d")
    p.on_new_request(producer)
    assert producer.num_prompt_tokens == 2048
    assert not producer.skip_reading_prefix_cache
    assert d.get_num_new_matched_tokens(consumer, 32) == (2016, True)


def test_long_prompt_keeps_history_needed_at_decode_start():
    p, d = scheduler("kv_producer"), scheduler("kv_consumer")
    producer, consumer = request(2048, "p"), request(2048, "d")
    p.on_new_request(producer)
    start, _ = d.get_num_new_matched_tokens(consumer, 0)
    # A 512-token left window must still exist when D starts at token2032.
    # Publishing all2048 and then rewinding D loses this producer history.
    core = SimpleNamespace(need_yoco_final_prompt_block_split=True, block_size=16)
    replay_start = Scheduler._yoco_final_prompt_block_split(core, consumer, 0, 2048)
    assert start == replay_start
    required_history = set(range(replay_start - 512, replay_start))
    retained_history = set(
        range(producer.num_prompt_tokens - 512, producer.num_prompt_tokens)
    )
    assert required_history <= retained_history
    assert Scheduler._yoco_final_prompt_block_split(core, producer, 0, 2032) == 2032


@pytest.mark.parametrize(
    "blocks,notify", [([], False), ([[], []], False), ([[0]], True), ([[], [4]], True)]
)
def test_empty_pull_releases_producer_without_waking_consumer(blocks, notify):
    worker = SimpleNamespace(finished_recving_reqs=set())
    pull = PullReqMeta(
        d_req_id="d",
        transfer_id="xfer",
        local_block_ids=blocks,
        remote_engine_id="p",
        remote_bootstrap_addr="http://p:1234",
        pull_tasks_count=2,
    )
    response = MooncakeXferResponse(
        status=MooncakeXferResponseStatus.FINISH, ok_reqs=["d"]
    )
    MooncakeConnectorWorker.process_pulling_result(worker, response, {"d": pull})
    assert not worker.finished_recving_reqs
    MooncakeConnectorWorker.process_pulling_result(worker, response, {"d": pull})
    assert pull.pull_tasks_count == 0
    assert worker.finished_recving_reqs == ({"d"} if notify else set())
