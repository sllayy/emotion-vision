import matplotlib
matplotlib.use("Agg")  # GUI açma, sadece bellekte işle

import matplotlib.pyplot as plt

def plot_emotions(probabilities, labels, filename="emotion_chart.png"):
    plt.figure(figsize=(8, 4))
    plt.bar(labels, probabilities, color='skyblue')
    plt.title("Duygu Tahmin Dağılımı")
    plt.xlabel("Duygular")
    plt.ylabel("Olasılık")
    plt.ylim([0, 1])
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
