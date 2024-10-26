import matplotlib.pyplot as plt


def display_history(model_history):
    """
    Flexibly supports different losses and metrics.
    Applies some heuristics to group the losses together onto a log plot, and the other metrics
    on a linear plot.
    :param model_history:
    """

    # identify "losses" vs other metrics
    loss_keys = [k for k in model_history.history.keys() if 'loss' in k or 'entropy' in k]
    metric_keys = [k for k in model_history.history.keys() if 'loss' not in k and 'entropy' not in k]

    plt.figure(figsize=(11, 3))
    if len(loss_keys) > 0:
        plt.subplot(1, 2, 1)
        plt.title("Loss")
        for key in loss_keys:
            plt.plot(model_history.epoch, model_history.history[key], label=key)
        plt.legend()
        plt.yscale('log')
        plt.xlabel('Epoch')

    if len(metric_keys) > 0:
        plt.subplot(1, 2, 2)
        plt.title("Metrics")
        for key in metric_keys:
            plt.plot(model_history.epoch, model_history.history[key], label=key)
        plt.legend()
        plt.xlabel('Epoch')

    plt.show()
