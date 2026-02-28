import os
import matplotlib.pyplot as plt

from master_table import MasterTable
from part1_dense import train_dense
from part2_cnn import train_cnn
from part3_optimizers import train_optimizer


# ===============================
# Create plots folder if missing
# ===============================
if not os.path.exists("plots"):
    os.makedirs("plots")

# ===============================
# Initialize Master Table
# ===============================
master = MasterTable()


# =====================================================
# =================== PART 1 ==========================
# =====================================================

print("\nRunning Part 1: Dense Networks\n")

dense_architectures = [
    [2, 8, 1],                 # 2-layer
    [2, 8, 8, 8, 8, 1],        # 5-layer
    [2, 8, 8, 8, 8, 8, 8, 8, 8, 1]  # 10-layer
]

for layers in dense_architectures:

    depth = len(layers) - 1

    train_acc, val_acc, test_acc, params = train_dense(
        layers,
        activation="relu",
        epochs=200,
        lr=0.01
    )

    master.add(
        model="Dense",
        depth=depth,
        activation="ReLU",
        optimizer="SGD",
        params=params,
        train_acc=train_acc,
        val_acc=val_acc,
        test_acc=test_acc
    )


# =====================================================
# =================== PART 2 ==========================
# =====================================================

print("\nRunning Part 2: CNN vs Dense\n")

train_acc, val_acc, test_acc, params = train_cnn(
    epochs=5,
    lr=0.01
)

master.add(
    model="CNN",
    depth=1,
    activation="ReLU",
    optimizer="SGD",
    params=params,
    train_acc=train_acc,
    val_acc=val_acc,
    test_acc=test_acc
)


# =====================================================
# =================== PART 3 ==========================
# =====================================================

print("\nRunning Part 3: Optimizer Comparison\n")

plt.figure()

optimizers = ["sgd", "momentum", "adam"]

for opt in optimizers:

    print(f"Training with {opt.upper()}")

    train_acc, val_acc, test_acc, params, losses = train_optimizer(
        opt,
        epochs=5,
        lr=0.01
    )

    master.add(
        model="CNN",
        depth=1,
        activation="ReLU",
        optimizer=opt.upper(),
        params=params,
        train_acc=train_acc,
        val_acc=val_acc,
        test_acc=test_acc
    )

    plt.plot(losses, label=opt.upper())

plt.title("Optimizer Convergence (CNN)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("plots/optimizer_comparison.png")
plt.close()


# =====================================================
# ================= FINAL OUTPUT ======================
# =====================================================

master.display()
master.save("master_table.csv")

print("\nAll experiments completed successfully.")
print("Master table saved as master_table.csv")
print("Plots saved inside 'plots/' folder.")