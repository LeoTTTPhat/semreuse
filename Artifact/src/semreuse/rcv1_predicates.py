"""Natural-language predicates over the RCV1-v2 topic hierarchy.

RCV1-v2 gives what 20 Newsgroups and AG News cannot: 804,414 documents, a
four-level topic hierarchy, and *multi-label* assignments (3.2 topics per story
on average, hierarchically expanded, so a story tagged ``C1511`` also carries
``C151``, ``C15``, ``CCAT``).  Two consequences shape the evaluation:

* containment is real and deep -- ``ext(C1511) subset ext(C151) subset
  ext(C15) subset ext(CCAT)`` holds exactly, giving multi-step reuse chains;
* disjointness has to be *earned*.  On a single-label corpus any two distinct
  leaves are automatically disjoint; here a story can be about corporate
  earnings and equity markets at once, so genuine ``OVERLAP`` -- the relation
  from which no rewrite rule can extract anything -- is the common case.

Topic descriptions are the standard RCV1-v2 topic names; codes we do not have
a confident gloss for are excluded from the predicate universe (they remain in
the corpus, so they still contribute to overlap between other predicates).
"""

from __future__ import annotations

import numpy as np

from semreuse.predicates import Predicate, PredicateUniverse, Workload

# code -> (noun phrase used in predicate text)
RCV1_TOPICS: dict[str, str] = {
    # Corporate / industrial
    "CCAT": "corporate or industrial news",
    "C11": "a company's strategy or business plans",
    "C12": "a legal or judicial matter involving a company",
    "C13": "the regulation of a company or an industry",
    "C14": "a share listing",
    "C15": "a company's financial performance",
    "C151": "a company's accounts or earnings",
    "C1511": "a company's annual results",
    "C152": "commentary or forecasts about a company's results",
    "C16": "corporate insolvency or a liquidity problem",
    "C17": "corporate funding or capital",
    "C171": "share capital",
    "C172": "a bond or debt issue",
    "C173": "a loan or a credit facility",
    "C174": "a credit rating",
    "C18": "a change in the ownership of a company",
    "C181": "a merger or an acquisition",
    "C182": "a transfer of assets",
    "C183": "a privatisation",
    "C21": "production or services output",
    "C22": "a new product or a new service",
    "C23": "research and development",
    "C24": "production capacity or facilities",
    "C31": "markets or marketing",
    "C311": "a domestic market",
    "C312": "an external or export market",
    "C313": "market share",
    "C32": "advertising or promotion",
    "C33": "a contract or an order",
    "C331": "a defence contract",
    "C34": "monopolies or competition policy",
    "C41": "company management",
    "C411": "a change of company management",
    "C42": "a company's labour force",
    # Economics
    "ECAT": "economics",
    "E11": "economic performance",
    "E12": "monetary or economic policy",
    "E121": "the money supply",
    "E13": "inflation or prices",
    "E131": "consumer prices",
    "E132": "wholesale prices",
    "E14": "consumer finance",
    "E141": "personal income",
    "E142": "consumer credit",
    "E143": "retail sales",
    "E21": "government finance",
    "E211": "government expenditure or revenue",
    "E212": "government borrowing",
    "E31": "economic output or capacity",
    "E311": "industrial production",
    "E312": "capacity utilization",
    "E313": "inventories",
    "E41": "employment or the labour market",
    "E411": "unemployment",
    "E51": "trade or foreign reserves",
    "E511": "the balance of payments",
    "E512": "merchandise trade",
    "E513": "foreign reserves",
    "E61": "housing starts",
    "E71": "leading economic indicators",
    # Government / social
    "GCAT": "government or social affairs",
    "GCRIM": "crime or law enforcement",
    "GDEF": "defence",
    "GDIP": "international relations",
    "GDIS": "a disaster or an accident",
    "GEDU": "education",
    "GENT": "arts, culture or entertainment",
    "GENV": "the environment or the natural world",
    "GFAS": "fashion",
    "GHEA": "health",
    "GJOB": "labour issues",
    "GOBIT": "an obituary",
    "GODD": "a human interest story",
    "GPOL": "domestic politics",
    "GPRO": "a biography or a profile of a person",
    "GREL": "religion",
    "GSCI": "science or technology",
    "GSPO": "sport",
    "GTOUR": "travel or tourism",
    "GVIO": "war or civil war",
    "GVOTE": "an election",
    "GWEA": "the weather",
    "GWELF": "welfare or social services",
    # Markets
    "MCAT": "financial markets",
    "M11": "equity markets",
    "M12": "bond markets",
    "M13": "money markets",
    "M131": "interbank markets",
    "M132": "foreign exchange markets",
    "M14": "commodity markets",
    "M141": "soft commodities",
    "M142": "metals trading",
    "M143": "energy markets",
}

TEMPLATES = [
    "The article is about {t}.",
    "This news story concerns {t}.",
    "The report covers {t}.",
]

UNION_TEMPLATE = "The article is about {a} or about {b}."


def build_rcv1_universe(corpus, templates: list[str] | None = None,
                        n_unions: int = 40, seed: int = 0
                        ) -> PredicateUniverse:
    """Predicates for every glossed topic code, plus cross-branch unions."""
    templates = templates or TEMPLATES
    known = [c for c in corpus.label_names if c in RCV1_TOPICS]
    preds: list[Predicate] = []
    for code in known:
        phrase = RCV1_TOPICS[code]
        for ti, tpl in enumerate(templates):
            preds.append(Predicate(tpl.format(t=phrase), frozenset([code]),
                                   name=f"topic:{code}#t{ti}"))
    # Unions across different roots: their extensions genuinely overlap other
    # predicates, which is the point of using a multi-label corpus.
    rng = np.random.default_rng(seed)
    roots = {c: corpus.root_of(c) for c in known}
    for _ in range(n_unions):
        a, b = rng.choice(len(known), size=2, replace=False)
        ca, cb = known[a], known[b]
        if roots[ca] == roots[cb]:
            continue
        preds.append(Predicate(
            UNION_TEMPLATE.format(a=RCV1_TOPICS[ca], b=RCV1_TOPICS[cb]),
            frozenset([ca, cb]), name=f"union:{ca}|{cb}"))
    return PredicateUniverse(corpus=corpus, predicates=preds)


def generate_drilldown_workload(universe, corpus, n_queries: int = 120,
                                overlap_rate: float = 0.8, seed: int = 0
                                ) -> Workload:
    """Sessions that walk the topic hierarchy the way an analyst would.

    A session picks a root topic and then drills down its descendants,
    occasionally rephrasing a predicate it already ran or reaching for a union
    across branches; with probability ``1 - overlap_rate`` a query opens a new
    session on an unrelated branch.
    """
    rng = np.random.default_rng(seed)
    by_code: dict[str, list[Predicate]] = {}
    unions: list[Predicate] = []
    for p in universe.predicates:
        if p.name.startswith("topic:"):
            by_code.setdefault(p.name.split(":")[1].split("#")[0], []).append(p)
        else:
            unions.append(p)
    codes = sorted(by_code)
    roots = [c for c in codes if corpus.parents.get(c) is None]
    queries: list[Predicate] = []
    session: list[str] = []

    def pick(xs):
        return xs[int(rng.integers(len(xs)))]

    while len(queries) < n_queries:
        if not session or rng.random() > overlap_rate:
            root = pick(roots) if roots else pick(codes)
            session = [root]
            queries.append(pick(by_code[root]))
            continue
        r = rng.random()
        if r < 0.55:                                   # drill down
            cur = pick(session)
            kids = [c for c in corpus.children_of(cur) if c in by_code]
            nxt = pick(kids) if kids else pick(session)
            session.append(nxt)
            queries.append(pick(by_code[nxt]))
        elif r < 0.85:                                 # rephrase / re-run
            queries.append(pick(by_code[pick(session)]))
        elif unions:                                   # cross-branch union
            queries.append(pick(unions))
        else:
            queries.append(pick(by_code[pick(session)]))
    return Workload(queries=queries[:n_queries], universe=universe,
                    params=dict(n_queries=n_queries,
                                overlap_rate=overlap_rate, seed=seed,
                                generator="rcv1-drilldown"))
