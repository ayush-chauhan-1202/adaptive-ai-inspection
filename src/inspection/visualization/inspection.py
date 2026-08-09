import matplotlib.pyplot as plt


def plot_inspection_result(
    image,
    score,
    threshold,
    prediction,
    ground_truth,
    output_path,
):
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(image, cmap="gray")

    title = (
        f"Score: {score:.3f} | "
        f"Threshold: {threshold:.3f}\n"
        f"Prediction: {prediction} | "
        f"Ground Truth: {ground_truth}"
    )

    ax.set_title(title)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)