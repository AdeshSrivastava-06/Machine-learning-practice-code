# Association rule mining with the Apriori algorithm on the Titanic data.
# Goal: find items that often appear together, and turn them into rules of the
# form IF A THEN B. Here every passenger is one "basket" of items, such as
# Sex=female, Class=1 and Survived=yes.
#
# Measures used for a rule IF A THEN B:
#   Support    : share of all baskets that contain both A and B
#   Confidence : of the baskets with A, the share that also contain B
#   Lift       : confidence divided by the overall share of B. A lift above 1
#                means A and B appear together more often than by chance.
#
# Part 1 builds Apriori from scratch so every step is visible.
# Part 2 does the same with the mlxtend library (pip install mlxtend).
# Part 3 compares the two results.
# Required file: titanic.csv with the standard Kaggle columns.

from itertools import combinations
import pandas as pd

CSV_PATH = "train.csv"
MIN_SUPPORT = 0.05
MIN_CONFIDENCE = 0.6

# Build the baskets 
df = pd.read_csv(CSV_PATH)
needed = ["Pclass", "Sex", "Age", "Embarked", "Survived"]
missing_cols = [c for c in needed if c not in df.columns]
if missing_cols:
    raise SystemExit(f"Columns not found in {CSV_PATH}: {missing_cols}")

df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 12, 59, 120], labels=["child", "adult", "senior"])

baskets = []
for _, row in df.iterrows():
    items = {
        f"Class={row['Pclass']}",
        f"Sex={row['Sex']}",
        f"Survived={'yes' if row['Survived'] == 1 else 'no'}",
    }
    # Passengers with an unknown age or port simply have no item for that column
    if pd.notna(row["AgeGroup"]):
        items.add(f"Age={row['AgeGroup']}")
    if pd.notna(row["Embarked"]):
        items.add(f"Port={row['Embarked']}")
    baskets.append(frozenset(items))

n = len(baskets)
print("Baskets:", n)
print("Example:", sorted(baskets[0]))
print(f"Minimum support: {MIN_SUPPORT}, minimum confidence: {MIN_CONFIDENCE}")
print()

# Part 1: Apriori from scratch 
def support_of(itemset):
    return sum(itemset <= basket for basket in baskets) / n

def apriori_scratch():
    all_items = sorted({item for basket in baskets for item in basket})
    frequent = {}
    current = {}
    for item in all_items:
        s = support_of(frozenset([item]))
        if s >= MIN_SUPPORT:
            current[frozenset([item])] = s
    frequent.update(current)
    print(f"Level 1: {len(all_items)} items, {len(current)} frequent")

    k = 2
    while current:
        # Join step: combine frequent itemsets of size k-1 into candidates of size k
        previous = list(current)
        candidates = {a | b for a in previous for b in previous if len(a | b) == k}
        # Prune step (Apriori property): every subset of a frequent itemset
        # must also be frequent, so candidates with an infrequent subset are dropped
        pruned = {c for c in candidates
                  if all(frozenset(sub) in current for sub in combinations(c, k - 1))}
        current = {}
        for c in pruned:
            s = support_of(c)
            if s >= MIN_SUPPORT:
                current[c] = s
        frequent.update(current)
        print(f"Level {k}: {len(candidates)} candidates, {len(pruned)} after pruning, "
              f"{len(current)} frequent")
        k += 1
    return frequent

def rules_scratch(frequent):
    rules = []
    for itemset, sup in frequent.items():
        if len(itemset) < 2:
            continue
        # Try every way of splitting the itemset into an IF part and a THEN part
        for size in range(1, len(itemset)):
            for antecedent in combinations(itemset, size):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                confidence = sup / frequent[antecedent]
                if confidence >= MIN_CONFIDENCE:
                    lift = confidence / frequent[consequent]
                    rules.append((antecedent, consequent, sup, confidence, lift))
    return pd.DataFrame(rules, columns=["antecedent", "consequent", "support", "confidence", "lift"])

print("Part 1: Apriori from scratch")
frequent_scratch = apriori_scratch()
rules_s = rules_scratch(frequent_scratch)
print(f"Frequent itemsets: {len(frequent_scratch)}, rules: {len(rules_s)}")

def show_rules(rules, title, count=8, sort_by="lift"):
    print()
    print(title)
    top = rules.sort_values(sort_by, ascending=False).head(count)
    for _, r in top.iterrows():
        print(f"IF {', '.join(sorted(r['antecedent']))} THEN {', '.join(sorted(r['consequent']))}"
              f"  (support {r['support']:.3f}, confidence {r['confidence']:.3f}, lift {r['lift']:.2f})")

show_rules(rules_s, "Top rules by lift")

survival = rules_s[rules_s["consequent"].apply(lambda c: c == frozenset(["Survived=yes"]))]
show_rules(survival, "Rules that lead to Survived=yes (sorted by confidence)", 5, sort_by="confidence")

# Part 2: the same with mlxtend 
print()
print("Part 2: Apriori with the mlxtend library")
try:
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules
except ImportError:
    print("mlxtend is not installed. Run: pip install mlxtend")
    raise SystemExit

encoder = TransactionEncoder()
encoded = pd.DataFrame(encoder.fit(baskets).transform(baskets), columns=encoder.columns_)
frequent_lib = apriori(encoded, min_support=MIN_SUPPORT, use_colnames=True)
try:
    rules_l = association_rules(frequent_lib, metric="confidence", min_threshold=MIN_CONFIDENCE)
except TypeError:
    rules_l = association_rules(frequent_lib, num_itemsets=len(encoded),
                                metric="confidence", min_threshold=MIN_CONFIDENCE)
print(f"Frequent itemsets: {len(frequent_lib)}, rules: {len(rules_l)}")
show_rules(rules_l.rename(columns={"antecedents": "antecedent", "consequents": "consequent"}),
           "Top rules by lift")

#  Part 3: comparison 
print()
print("Part 3: comparison")
same_itemsets = {frozenset(x) for x in frequent_lib["itemsets"]} == set(frequent_scratch)
same_rules = {(a, c) for a, c in zip(rules_l["antecedents"], rules_l["consequents"])} == \
             {(a, c) for a, c in zip(rules_s["antecedent"], rules_s["consequent"])}
print(f"{'':22s} {'Scratch':>8s} {'mlxtend':>8s}")
print(f"{'Frequent itemsets':22s} {len(frequent_scratch):8d} {len(frequent_lib):8d}")
print(f"{'Rules':22s} {len(rules_s):8d} {len(rules_l):8d}")
print("Same itemsets found:", same_itemsets)
print("Same rules found:", same_rules)

# Summary
best = rules_s.sort_values("lift", ascending=False).iloc[0]
print()
print("Summary:")
print(f"The strongest rule by lift is IF {', '.join(sorted(best['antecedent']))} THEN "
      f"{', '.join(sorted(best['consequent']))} with lift {best['lift']:.2f}.")
if same_itemsets and same_rules:
    print("The scratch version and mlxtend found exactly the same itemsets and rules.")
else:
    print("The two versions gave different results, so the code should be checked.")
print("A high confidence alone can be misleading when the THEN item is very common,")
print("so lift should be checked as well. Association rules show that items appear")
print("together, not that one item causes the other.")
