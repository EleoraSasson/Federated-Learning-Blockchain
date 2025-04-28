# gqia_three_plots.py
import copy, torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from utils   import load_mnist_data, create_client_data
from models  import Net
from clients import FederatedClient
from server  import FederatedServer

# ---------------- configuration ----------------
NUM_CLIENTS   = 4
MALICIOUS_ID  = 0
ROUNDS        = 5
LOCAL_EPOCHS  = 5
REWARD_POOL   = 100.0                # tokens / round
# ----------------------------------------------

# data ------------------------------------------------
train_ds, test_ds = load_mnist_data()
val_ds, test_ds   = random_split(test_ds,
                                 [len(test_ds)//2,
                                  len(test_ds)-len(test_ds)//2])

test_loader = DataLoader(test_ds, batch_size=128)
val_loader  = DataLoader(val_ds,  batch_size=128)

client_loaders, _, client_sizes = create_client_data(train_ds, NUM_CLIENTS)

# server + clients ------------------------------------
global_model = Net(input_dim=28*28, hidden_dim=128, output_dim=10)
server = FederatedServer(global_model,
                         test_loader=test_loader,
                         validation_loader=val_loader,
                         blockchain_enabled=False)

clients = [FederatedClient(cid,
                           model=copy.deepcopy(global_model),
                           data_loader=client_loaders[cid],
                           epochs=LOCAL_EPOCHS)
           for cid in range(NUM_CLIENTS)]

# trackers --------------------------------------------
acc      = []                                # global accuracy per round
scores   = {cid: [] for cid in range(NUM_CLIENTS)}
rewards  = {cid: [] for cid in range(NUM_CLIENTS)}

# federated loop --------------------------------------
for rnd in range(1, ROUNDS+1):
    server.start_round(f"rnd{rnd}")
    local_models = []

    for cid, cl in enumerate(clients):
        if cid == MALICIOUS_ID:
            with torch.no_grad():
                for p in cl.model.parameters():
                    p.data = torch.randn_like(p)    # random update
        else:
            cl.train()
        local_models.append(copy.deepcopy(cl.model))

    server.aggregate(local_models, client_sizes,
                     client_addresses=[f"c{cid}" for cid in range(NUM_CLIENTS)])
    gqia = server.contribution_scores

    for cid, sc in enumerate(gqia):
        scores[cid].append(sc)

    tot = sum(gqia)
    r_vec = [(sc/tot)*REWARD_POOL if tot else 0 for sc in gqia]
    for cid, rw in enumerate(r_vec):
        rewards[cid].append(rw)

    acc.append(server.evaluate())
    new_global = server.get_global_model()
    for cl in clients:
        cl.update_model(new_global)

# ---------------- plotting  (NO legends) -------------
round_axis = list(range(1, ROUNDS+1))

# 1. accuracy plot
plt.figure(figsize=(4,3))
plt.plot(round_axis, acc, marker='o')
plt.title("Global Model Accuracy")
plt.xlabel("Round"); plt.ylabel("Accuracy (%)"); plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_curve.png", dpi=150)
plt.close()

# 2. GQIA score plot
plt.figure(figsize=(4,3))
for cid in scores:
    plt.plot(round_axis, scores[cid], marker='o')
plt.title("Client Contributions (GQIA)")
plt.xlabel("Round"); plt.ylabel("Contribution Score"); plt.grid(True)
plt.tight_layout()
plt.savefig("gqia_scores.png", dpi=150)
plt.close()

# 3. reward plot
plt.figure(figsize=(4,3))
for cid in rewards:
    plt.plot(round_axis, rewards[cid], marker='o')
plt.title("Client Rewards per Round")
plt.xlabel("Round"); plt.ylabel("Tokens"); plt.grid(True)
plt.tight_layout()
plt.savefig("client_rewards.png", dpi=150)
plt.close()

print("✅  Saved: accuracy_curve.png, gqia_scores.png, client_rewards.png")
