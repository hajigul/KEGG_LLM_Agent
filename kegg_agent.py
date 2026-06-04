"""
KEGG50k Relation Discovery Agent
---------------------------------
Predicts the relation between a head and tail entity (drug-relation discovery).

Usage:
    python kegg_agent.py --train          # train the model
    python kegg_agent.py --predict H00787 hsa04080
    python kegg_agent.py --interactive    # interactive query loop

Requires: torch, numpy
    pip install torch numpy
"""

import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DATA_DIR = r"C:\Users\Haji\Documents\kegg_llm_agent\data"
TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
VALID_FILE = os.path.join(DATA_DIR, "valid.txt")
TEST_FILE  = os.path.join(DATA_DIR, "test.txt")

MODEL_PATH = os.path.join(DATA_DIR, "relation_agent.pt")
MAP_PATH   = os.path.join(DATA_DIR, "mappings.pkl")

EMB_DIM    = 200
EPOCHS     = 50
BATCH_SIZE = 1024
LR         = 1e-3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------
def read_triplets(path):
    """Read a tab-separated triplet file: head \t relation \t tail."""
    triplets = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                parts = line.split()  # fallback: whitespace split
            if len(parts) != 3:
                continue
            h, r, t = parts
            triplets.append((h, r, t))
    return triplets


def build_mappings(all_triplets):
    """Map every entity and relation to an integer id."""
    entities, relations = set(), set()
    for h, r, t in all_triplets:
        entities.add(h)
        entities.add(t)
        relations.add(r)
    ent2id = {e: i for i, e in enumerate(sorted(entities))}
    rel2id = {r: i for i, r in enumerate(sorted(relations))}
    return ent2id, rel2id


class RelationDataset(Dataset):
    """Yields (head_id, tail_id, relation_id) for relation classification."""
    def __init__(self, triplets, ent2id, rel2id):
        self.data = []
        for h, r, t in triplets:
            if h in ent2id and t in ent2id and r in rel2id:
                self.data.append((ent2id[h], ent2id[t], rel2id[r]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        h, t, r = self.data[idx]
        return (torch.tensor(h), torch.tensor(t), torch.tensor(r))


# ----------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------
class RelationAgentModel(nn.Module):
    """
    Learns entity embeddings, then classifies the relation between a
    head and tail entity from their embeddings.
    """
    def __init__(self, n_entities, n_relations, emb_dim=EMB_DIM):
        super().__init__()
        self.ent_emb = nn.Embedding(n_entities, emb_dim)
        nn.init.xavier_uniform_(self.ent_emb.weight)

        # classifier head: takes [head, tail, head*tail, head-tail]
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim * 4, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(emb_dim, n_relations),
        )

    def forward(self, h_idx, t_idx):
        h = self.ent_emb(h_idx)
        t = self.ent_emb(t_idx)
        feat = torch.cat([h, t, h * t, h - t], dim=-1)
        return self.classifier(feat)   # logits over relations


# ----------------------------------------------------------------------
# TRAINING
# ----------------------------------------------------------------------
def train():
    print(f"Device: {DEVICE}")
    print("Loading triplets...")
    train_tr = read_triplets(TRAIN_FILE)
    valid_tr = read_triplets(VALID_FILE) if os.path.exists(VALID_FILE) else []
    test_tr  = read_triplets(TEST_FILE)  if os.path.exists(TEST_FILE)  else []

    ent2id, rel2id = build_mappings(train_tr + valid_tr + test_tr)
    id2rel = {i: r for r, i in rel2id.items()}
    print(f"Entities: {len(ent2id)}  Relations: {len(rel2id)}  "
          f"Train triplets: {len(train_tr)}")

    with open(MAP_PATH, "wb") as f:
        pickle.dump({"ent2id": ent2id, "rel2id": rel2id}, f)

    train_ds = RelationDataset(train_tr, ent2id, rel2id)
    valid_ds = RelationDataset(valid_tr, ent2id, rel2id) if valid_tr else None
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = RelationAgentModel(len(ent2id), len(rel2id)).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for h, t, r in train_dl:
            h, t, r = h.to(DEVICE), t.to(DEVICE), r.to(DEVICE)
            optim.zero_grad()
            logits = model(h, t)
            loss = loss_fn(logits, r)
            loss.backward()
            optim.step()
            total_loss += loss.item() * h.size(0)
        avg = total_loss / len(train_ds)

        msg = f"Epoch {epoch:3d}  loss {avg:.4f}"
        if valid_ds:
            msg += f"  val_acc {evaluate(model, valid_ds):.4f}"
        print(msg)

    if test_tr:
        test_ds = RelationDataset(test_tr, ent2id, rel2id)
        print(f"\nTest accuracy: {evaluate(model, test_ds):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


@torch.no_grad()
def evaluate(model, dataset):
    model.eval()
    dl = DataLoader(dataset, batch_size=BATCH_SIZE)
    correct = total = 0
    for h, t, r in dl:
        h, t, r = h.to(DEVICE), t.to(DEVICE), r.to(DEVICE)
        pred = model(h, t).argmax(dim=-1)
        correct += (pred == r).sum().item()
        total += r.size(0)
    return correct / max(total, 1)


# ----------------------------------------------------------------------
# THE AGENT
# ----------------------------------------------------------------------
class RelationDiscoveryAgent:
    """Given a head and tail entity, predicts the relation between them."""
    def __init__(self):
        with open(MAP_PATH, "rb") as f:
            maps = pickle.load(f)
        self.ent2id = maps["ent2id"]
        self.rel2id = maps["rel2id"]
        self.id2rel = {i: r for r, i in self.rel2id.items()}

        self.model = RelationAgentModel(len(self.ent2id), len(self.rel2id))
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        self.model.to(DEVICE).eval()

    @torch.no_grad()
    def predict(self, head, tail, top_k=5):
        if head not in self.ent2id:
            return f"Unknown head entity: {head}"
        if tail not in self.ent2id:
            return f"Unknown tail entity: {tail}"

        h = torch.tensor([self.ent2id[head]]).to(DEVICE)
        t = torch.tensor([self.ent2id[tail]]).to(DEVICE)
        probs = torch.softmax(self.model(h, t), dim=-1).squeeze(0)
        top_k = min(top_k, probs.numel())
        vals, idxs = torch.topk(probs, top_k)

        return [(self.id2rel[i.item()], round(v.item(), 4))
                for v, i in zip(vals, idxs)]


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="KEGG50k Relation Discovery Agent")
    parser.add_argument("--train", action="store_true", help="train the model")
    parser.add_argument("--predict", nargs=2, metavar=("HEAD", "TAIL"),
                        help="predict relation between HEAD and TAIL")
    parser.add_argument("--interactive", action="store_true",
                        help="interactive prediction loop")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    if args.train:
        train()
    elif args.predict:
        agent = RelationDiscoveryAgent()
        head, tail = args.predict
        result = agent.predict(head, tail, top_k=args.top_k)
        print(f"\nHead: {head}   Tail: {tail}")
        print("Predicted relations (relation, confidence):")
        if isinstance(result, str):
            print(" ", result)
        else:
            for rel, conf in result:
                print(f"  {rel:<25} {conf}")
    elif args.interactive:
        agent = RelationDiscoveryAgent()
        print("Interactive mode. Enter: HEAD TAIL   (or 'quit')")
        while True:
            line = input("> ").strip()
            if line.lower() in ("quit", "exit", "q"):
                break
            parts = line.split()
            if len(parts) != 2:
                print("Please enter exactly: HEAD TAIL")
                continue
            result = agent.predict(parts[0], parts[1], top_k=args.top_k)
            if isinstance(result, str):
                print(" ", result)
            else:
                for rel, conf in result:
                    print(f"  {rel:<25} {conf}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()