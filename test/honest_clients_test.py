import copy, os, random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import torch

# ---------------- project imports ----------------
from utils   import load_mnist_data, create_client_data
from models  import Net
from clients import FederatedClient
from server  import FederatedServer
# -------------------------------------------------

# ------------------- CONFIG ----------------------
NUM_CLIENTS   = 4
LOCAL_EPOCHS  = 5
ROUNDS        = 5
SEED          = 1234
# -------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ------------ data -------------------------------
train_ds, test_ds = load_mnist_data()
val_size          = len(test_ds) // 2
val_ds, test_ds   = random_split(test_ds, [val_size, len(test_ds) - val_size])
test_loader       = DataLoader(test_ds,  batch_size=128)
val_loader        = DataLoader(val_ds,   batch_size=128)

client_loaders, _, client_sizes = create_client_data(train_ds, NUM_CLIENTS)

# ------------ global model & server --------------
global_model = Net(input_dim=28*28, hidden_dim=128, output_dim=10)
server = FederatedServer(
    global_model,
    test_loader=test_loader,
    validation_loader=val_loader,
    blockchain_enabled=False   # off-chain
)

# ------------ build honest clients ---------------
clients = []
for cid in range(NUM_CLIENTS):
    clients.append(
        FederatedClient(
            client_id=cid,
            model=copy.deepcopy(global_model),
            data_loader=client_loaders[cid],
            epochs=LOCAL_EPOCHS
        )
    )

# ------------ history containers -----------------
hist_acc   = []                                # global accuracy per round
hist_score = {cid: [] for cid in range(NUM_CLIENTS)}  # GQIA scores
hist_rw    = {cid: [] for cid in range(NUM_CLIENTS)}  # cumulative rewards

# ------------ FL loop ----------------------------
for rnd in range(1, ROUNDS + 1):
    print(f"\n========= ROUND {rnd} =========")
    server.start_round(f"key_{rnd}")

    # --- local training ---
    client_models = []
    for client in clients:
        print(f"Training Client {client.client_id}")
        client.train()
        client_models.append(copy.deepcopy(client.model))

    # --- aggregation & GQIA scoring --------------
    addr_list = [f"c{cid}" for cid in range(NUM_CLIENTS)]
    server.aggregate(client_models, client_sizes, addr_list)

    # record GQIA scores
    for cid, score in enumerate(server.contribution_scores):
        hist_score[cid].append(score)

    # --- evaluate global --------------------------
    acc = server.evaluate()
    hist_acc.append(acc)

    # --- distribute & record rewards -------------
    server.finalize_round(reward_per_round=50.0)
    for cid, addr in enumerate(addr_list):
        bal = server.token_ledger.get_token_balance(addr)
        hist_rw[cid].append(bal)

# ------------- PLOTTING --------------------------
os.makedirs("Plots", exist_ok=True)
round_axis = np.arange(1, ROUNDS + 1)

# 1) global accuracy
plt.figure(figsize=(4,3))
plt.plot(round_axis, hist_acc, marker='o')
plt.title("Global Model Accuracy")
plt.xlabel("Round")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig("Plots/new_accuracy_curve.png")
plt.close()

# 2) GQIA scores
plt.figure(figsize=(4,3))
for cid, scores in hist_score.items():
    plt.plot(round_axis, scores, marker='o', label=f"Client {cid}")
plt.title("Client Contributions (GQIA)")
plt.xlabel("Round")
plt.ylabel("Contribution Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Plots/new_gqia_scores.png")
plt.close()

# 3) rewards per round (delta of cumulative balances)
plt.figure(figsize=(4,3))
for cid, balances in hist_rw.items():
    # compute actual tokens earned each round
    rewards_per_round = [balances[0]] + [
        balances[i] - balances[i-1]
        for i in range(1, len(balances))
    ]
    plt.plot(round_axis, rewards_per_round, marker='o', label=f"Client {cid}")
plt.title("Client Rewards per Round")
plt.xlabel("Round")
plt.ylabel("Tokens")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Plots/new_client_rewards.png")
plt.close()

print("✅  Saved plots to Plots/new_accuracy_curve.png, Plots/new_gqia_scores.png, Plots/new_client_rewards.png")






