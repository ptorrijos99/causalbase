"""Background knowledge: adjacencies the search must keep or must not create."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Integral

from causalbase.graph import DAG

__all__ = ["Knowledge"]

PairLike = tuple[int | str, int | str]


@dataclass(frozen=True)
class Knowledge:
    """Adjacencies fixed in advance, by domain knowledge or by design.

    Both kinds constrain whether two variables are adjacent, not the direction
    of the edge between them: a search over equivalence classes decides
    directions from the data, and a required edge keeps whichever direction the
    data favour.

    Attributes
    ----------
    required : frozenset of frozenset of int
        Pairs that are always adjacent.
    forbidden : frozenset of frozenset of int
        Pairs that are never adjacent.
    """

    required: frozenset[frozenset[int]] = frozenset()
    forbidden: frozenset[frozenset[int]] = frozenset()

    @classmethod
    def from_pairs(
        cls,
        required: Iterable[PairLike] = (),
        forbidden: Iterable[PairLike] = (),
        names: Sequence[str] = (),
    ) -> Knowledge:
        """Build knowledge from pairs of variable names or column indices.

        Parameters
        ----------
        required, forbidden : iterable of pairs
            Pairs of names in `names` or of column indices into it.
        names : sequence of str
            The variables, in column order.

        Raises
        ------
        ValueError
            If a name is unknown, an index is not a column of `names`, a pair
            joins a variable to itself, or a pair is both required and
            forbidden.
        """
        index = {name: i for i, name in enumerate(names)}

        def resolve(pairs: Iterable[PairLike]) -> frozenset[frozenset[int]]:
            out = set()
            for pair in pairs:
                ends = []
                for end in pair:
                    if isinstance(end, str):
                        if end not in index:
                            raise ValueError(f"unknown variable {end!r}")
                        ends.append(index[end])
                    elif isinstance(end, Integral) and not isinstance(end, bool):
                        if not 0 <= int(end) < len(names):
                            raise ValueError(f"no variable at index {end!r}")
                        ends.append(int(end))
                    else:
                        raise ValueError(f"{end!r} is neither a variable name nor an index")
                if len(ends) != 2 or ends[0] == ends[1]:
                    raise ValueError(f"{pair!r} is not a pair of two distinct variables")
                out.add(frozenset(ends))
            return frozenset(out)

        knowledge = cls(resolve(required), resolve(forbidden))
        both = knowledge.required & knowledge.forbidden
        if both:
            pairs = sorted(tuple(sorted(p)) for p in both)
            raise ValueError(f"pairs both required and forbidden: {pairs}")
        return knowledge

    def __bool__(self) -> bool:
        return bool(self.required or self.forbidden)

    def may_add(self, x: int, y: int) -> bool:
        """Return whether `x` and `y` may become adjacent."""
        return frozenset((x, y)) not in self.forbidden

    def may_remove(self, x: int, y: int) -> bool:
        """Return whether the edge between `x` and `y` may be removed."""
        return frozenset((x, y)) not in self.required

    def impose(self, dag: DAG) -> DAG:
        """Return `dag` without its forbidden adjacencies and with every required one.

        A missing required edge is added in the direction of a topological
        order of `dag`, so the result stays acyclic.
        """
        position = {v: i for i, v in enumerate(dag.topological_order())}
        edges = {e for e in dag.edges() if frozenset(e) not in self.forbidden}
        present = {frozenset(e) for e in edges}
        for pair in self.required - present:
            u, v = sorted(pair, key=position.__getitem__)
            edges.add((u, v))
        return DAG.from_edges(dag.num_nodes, sorted(edges))
