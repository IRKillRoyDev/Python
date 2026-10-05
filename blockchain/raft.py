"""
Raft consensus, simulated in one process.

A cluster of nodes agrees on an ordered log of commands. One leader accepts
new commands, followers copy that log, and a command is committed only after
a majority has stored it. Integer ticks stand in for the network and the
election clock, so the same sequence of ticks always elects the same leader.

https://en.wikipedia.org/wiki/Raft_(algorithm)
https://raft.github.io/raft.pdf

Leader election and log replication are included. Membership changes and
snapshotting are not.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _Entry:
    term: int
    command: str | None


@dataclass(frozen=True)
class _RequestVote:
    term: int
    candidate_id: str
    last_log_index: int
    last_log_term: int


@dataclass(frozen=True)
class _VoteReply:
    term: int
    voter_id: str
    granted: bool


@dataclass(frozen=True)
class _AppendEntries:
    term: int
    leader_id: str
    prev_log_index: int
    prev_log_term: int
    entries: tuple[_Entry, ...]
    leader_commit: int


@dataclass(frozen=True)
class _AppendReply:
    term: int
    follower_id: str
    success: bool
    match_index: int


_Message = _RequestVote | _VoteReply | _AppendEntries | _AppendReply


class _Volatile:
    def __init__(self) -> None:
        self.commit_index = 0
        self.role = "follower"
        self.election_elapsed = 0
        self.votes: set[str] = set()
        self.next_index: dict[str, int] = {}
        self.match_index: dict[str, int] = {}


class _Node:
    def __init__(self, node_id: str, peers: list[str], election_timeout: int) -> None:
        self.node_id = node_id
        self.peers = peers
        self.election_timeout = election_timeout
        self.term = 0
        self.voted_for: str | None = None
        self.log: list[_Entry] = [_Entry(0, None)]
        self.volatile = _Volatile()

    def committed_commands(self) -> list[str]:
        commands: list[str] = []
        for entry in self.log[1 : self.volatile.commit_index + 1]:
            if entry.command is not None:
                commands.append(entry.command)
        return commands

    def append_command(self, command: str) -> None:
        self.log.append(_Entry(self.term, command))

    def handle(self, message: _Message) -> list[tuple[str, _Message]]:
        if isinstance(message, _RequestVote):
            return self._on_request_vote(message)
        if isinstance(message, _VoteReply):
            return self._on_vote_reply(message)
        if isinstance(message, _AppendEntries):
            return self._on_append(message)
        return self._on_append_reply(message)

    def tick(self) -> list[tuple[str, _Message]]:
        if self.volatile.role == "leader":
            return self._heartbeats()
        self.volatile.election_elapsed += 1
        if self.volatile.election_elapsed >= self.election_timeout:
            return self._start_election()
        return []

    def _majority(self, count: int) -> bool:
        cluster_size = len(self.peers) + 1
        return count >= cluster_size // 2 + 1

    def _observe_term(self, term: int) -> None:
        if term > self.term:
            self.term = term
            self.voted_for = None
            self.volatile.role = "follower"
            self.volatile.votes.clear()

    def _log_is_current(self, last_index: int, last_term: int) -> bool:
        own_index = len(self.log) - 1
        own_term = self.log[own_index].term
        if last_term != own_term:
            return last_term > own_term
        return last_index >= own_index

    def _start_election(self) -> list[tuple[str, _Message]]:
        self.volatile.role = "candidate"
        self.term += 1
        self.voted_for = self.node_id
        self.volatile.votes = {self.node_id}
        self.volatile.election_elapsed = 0
        last_index = len(self.log) - 1
        request = _RequestVote(
            self.term, self.node_id, last_index, self.log[last_index].term
        )
        if self._majority(1):
            self._become_leader()
            return []
        return [(peer, request) for peer in self.peers]

    def _become_leader(self) -> None:
        self.volatile.role = "leader"
        last = len(self.log)
        self.volatile.next_index = dict.fromkeys(self.peers, last)
        self.volatile.match_index = dict.fromkeys(self.peers, 0)

    def _on_request_vote(self, message: _RequestVote) -> list[tuple[str, _Message]]:
        self._observe_term(message.term)
        granted = False
        vote_open = self.voted_for in (None, message.candidate_id)
        log_ok = self._log_is_current(message.last_log_index, message.last_log_term)
        if message.term == self.term and vote_open and log_ok:
            self.voted_for = message.candidate_id
            self.volatile.election_elapsed = 0
            granted = True
        reply = _VoteReply(self.term, self.node_id, granted)
        return [(message.candidate_id, reply)]

    def _on_vote_reply(self, message: _VoteReply) -> list[tuple[str, _Message]]:
        self._observe_term(message.term)
        if self.volatile.role != "candidate" or message.term != self.term:
            return []
        if message.granted:
            self.volatile.votes.add(message.voter_id)
        if self._majority(len(self.volatile.votes)):
            self._become_leader()
        return []

    def _on_append(self, message: _AppendEntries) -> list[tuple[str, _Message]]:
        self._observe_term(message.term)
        if message.term < self.term:
            reply = _AppendReply(self.term, self.node_id, False, 0)
            return [(message.leader_id, reply)]
        self.volatile.role = "follower"
        self.volatile.election_elapsed = 0
        if not self._prev_matches(message):
            reply = _AppendReply(self.term, self.node_id, False, 0)
            return [(message.leader_id, reply)]
        self._store_entries(message)
        if message.leader_commit > self.volatile.commit_index:
            self.volatile.commit_index = min(message.leader_commit, len(self.log) - 1)
        match_index = message.prev_log_index + len(message.entries)
        reply = _AppendReply(self.term, self.node_id, True, match_index)
        return [(message.leader_id, reply)]

    def _prev_matches(self, message: _AppendEntries) -> bool:
        if message.prev_log_index >= len(self.log):
            return False
        return self.log[message.prev_log_index].term == message.prev_log_term

    def _store_entries(self, message: _AppendEntries) -> None:
        index = message.prev_log_index + 1
        for entry in message.entries:
            if index < len(self.log) and self.log[index].term != entry.term:
                del self.log[index:]
            if index >= len(self.log):
                self.log.append(entry)
            index += 1

    def _on_append_reply(self, message: _AppendReply) -> list[tuple[str, _Message]]:
        self._observe_term(message.term)
        if self.volatile.role != "leader" or message.term != self.term:
            return []
        if message.success:
            known = self.volatile.match_index[message.follower_id]
            if message.match_index > known:
                self.volatile.match_index[message.follower_id] = message.match_index
                self.volatile.next_index[message.follower_id] = message.match_index + 1
            self._advance_commit()
        else:
            current = self.volatile.next_index[message.follower_id]
            self.volatile.next_index[message.follower_id] = max(1, current - 1)
        return []

    def _advance_commit(self) -> None:
        for index in range(len(self.log) - 1, self.volatile.commit_index, -1):
            if self.log[index].term != self.term:
                continue
            matched = 1
            for peer_match in self.volatile.match_index.values():
                if peer_match >= index:
                    matched += 1
            if self._majority(matched):
                self.volatile.commit_index = index
                return

    def _heartbeats(self) -> list[tuple[str, _Message]]:
        outgoing: list[tuple[str, _Message]] = []
        for peer in self.peers:
            next_index = self.volatile.next_index[peer]
            prev_index = next_index - 1
            entries = tuple(self.log[next_index:])
            message = _AppendEntries(
                self.term,
                self.node_id,
                prev_index,
                self.log[prev_index].term,
                entries,
                self.volatile.commit_index,
            )
            outgoing.append((peer, message))
        return outgoing


class Cluster:
    """
    Step a Raft cluster one tick at a time.

    The first node times out first, so it wins the election when nobody else
    has started one.

    >>> cluster = Cluster(["n0", "n1", "n2"])
    >>> cluster.propose("too-soon")
    False
    >>> for _ in range(8):
    ...     cluster.tick()
    >>> cluster.leader()
    'n0'
    >>> cluster.propose("x")
    True
    >>> cluster.propose("y")
    True
    >>> for _ in range(4):
    ...     cluster.tick()
    >>> cluster.committed()
    {'n0': ['x', 'y'], 'n1': ['x', 'y'], 'n2': ['x', 'y']}
    """

    def __init__(self, node_ids: list[str]) -> None:
        if len(node_ids) < 1 or len(set(node_ids)) != len(node_ids):
            msg = "node ids must be unique and non-empty"
            raise ValueError(msg)
        self.nodes: dict[str, _Node] = {}
        for index, node_id in enumerate(node_ids):
            peers = [other for other in node_ids if other != node_id]
            self.nodes[node_id] = _Node(node_id, peers, election_timeout=5 + 10 * index)
        self._inbox: dict[str, list[_Message]] = {node_id: [] for node_id in node_ids}

    def tick(self) -> None:
        """Deliver the previous tick's messages, then advance every node."""
        pending = {node_id: messages[:] for node_id, messages in self._inbox.items()}
        for messages in self._inbox.values():
            messages.clear()
        for node_id, messages in pending.items():
            for message in messages:
                for dest, reply in self.nodes[node_id].handle(message):
                    self._inbox[dest].append(reply)
        for node in self.nodes.values():
            for dest, message in node.tick():
                self._inbox[dest].append(message)

    def leader(self) -> str | None:
        """Return the only leader, or ``None`` when there is not exactly one."""
        leaders = [
            node_id
            for node_id, node in self.nodes.items()
            if node.volatile.role == "leader"
        ]
        if len(leaders) == 1:
            return leaders[0]
        return None

    def propose(self, command: str) -> bool:
        """Append ``command`` on the current leader. Return whether that happened."""
        leader_id = self.leader()
        if leader_id is None:
            return False
        self.nodes[leader_id].append_command(command)
        return True

    def committed(self) -> dict[str, list[str]]:
        """Return the commands each node has committed, in log order."""
        return {
            node_id: node.committed_commands() for node_id, node in self.nodes.items()
        }


if __name__ == "__main__":
    import doctest

    doctest.testmod()
