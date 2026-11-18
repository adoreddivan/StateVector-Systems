from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Protocol

import grpc
import structlog
from jito_searcher_client.async_searcher import AsyncSearcherInterceptor
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction

from jitokit._pb import bundle_pb2, packet_pb2, searcher_pb2, searcher_pb2_grpc

log = structlog.get_logger(__name__)

_LANDED = frozenset({"processed", "finalized"})
_TERMINAL = _LANDED | {"rejected", "dropped"}


def _to_packet(tx: VersionedTransaction) -> packet_pb2.Packet:
    raw = bytes(tx)
    return packet_pb2.Packet(data=raw, meta=packet_pb2.Meta(size=len(raw)))


async def _open_channel(block_engine: str, auth: Keypair) -> searcher_pb2_grpc.SearcherServiceStub:
    interceptor = AsyncSearcherInterceptor(block_engine, auth)
    await interceptor.authenticate_if_needed()
    channel = grpc.aio.secure_channel(
        block_engine,
        grpc.ssl_channel_credentials(),
        interceptors=[interceptor],
    )
    channel._unary_stream_interceptors.append(interceptor)
    return searcher_pb2_grpc.SearcherServiceStub(channel)


class ResultHook(Protocol):
    def __call__(self, uuid: str, outcome: str) -> None: ...


@dataclass(slots=True)
class Tally:
    sent: int = 0
    outcomes: Counter[str] = field(default_factory=Counter)
    pending: set[str] = field(default_factory=set)

    @property
    def landed(self) -> int:
        return sum(self.outcomes[o] for o in _LANDED)

    @property
    def land_rate(self) -> float:
        total = self.outcomes.total()
        return self.landed / total if total else 0.0


@dataclass(slots=True)
class Session:
    client: searcher_pb2_grpc.SearcherServiceStub
    tip_accounts: tuple[str, ...]
    on_result: ResultHook | None = None
    tally: Tally = field(default_factory=Tally)

    @property
    def land_rate(self) -> float:
        return self.tally.land_rate

    async def send(self, txs: Sequence[VersionedTransaction]) -> str:
        bundle = bundle_pb2.Bundle(header=None, packets=[_to_packet(tx) for tx in txs])
        resp = await self.client.SendBundle(searcher_pb2.SendBundleRequest(bundle=bundle))
        self.tally.sent += 1
        self.tally.pending.add(resp.uuid)
        log.debug("bundle sent", uuid=resp.uuid, txs=len(txs))
        return resp.uuid

    def _settle(self, r: bundle_pb2.BundleResult) -> None:
        outcome = r.WhichOneof("result")
        if r.bundle_id not in self.tally.pending or outcome not in _TERMINAL:
            return
        self.tally.pending.discard(r.bundle_id)
        self.tally.outcomes[outcome] += 1
        log.info(
            "bundle settled",
            uuid=r.bundle_id,
            outcome=outcome,
            landed=self.tally.landed,
            sent=self.tally.sent,
            land_rate=round(self.land_rate, 3),
        )
        if self.on_result is not None:
            self.on_result(r.bundle_id, outcome)

    async def _listen(self) -> None:
        async for r in self.client.SubscribeBundleResults(searcher_pb2.SubscribeBundleResultsRequest()):
            self._settle(r)


@asynccontextmanager
async def searcher(
    block_engine: str,
    auth: Keypair,
    on_result: ResultHook | None = None,
) -> AsyncIterator[Session]:
    client = await _open_channel(block_engine, auth)
    tips = await client.GetTipAccounts(searcher_pb2.GetTipAccountsRequest())
    session = Session(client, tuple(tips.accounts), on_result)
    async with asyncio.TaskGroup() as tg:
        listener = tg.create_task(session._listen(), name="jito-results")
        log.info("searcher ready", host=block_engine, signer=str(auth.pubkey()), tips=len(session.tip_accounts))
        try:
            yield session
        finally:
            listener.cancel()
            log.info(
                "searcher closed",
                sent=session.tally.sent,
                landed=session.tally.landed,
                unresolved=len(session.tally.pending),
                land_rate=round(session.land_rate, 3),
            )
