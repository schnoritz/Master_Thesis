class EarlyStopping():

    def __init__(self, patients):
        self.patients = patients
        self.stopping = False

    def __call__(self, epochs):

        if len(epochs) <= self.patients:
            return

        else:
            losses = []
            for epoch in epochs:
                losses.append(epoch['validation_loss'])

            if losses[-self.patients:].index(min(losses[-self.patients:])) == 0:
                self.stopping = True


if __name__ == "__main__":

    stopping = EarlyStopping(patients=20)

    loss = list(range(20, 4, -1))
    loss.extend(list(range(4, 10)))
    loss.append(2)
    loss.extend(list(range(10, 50)))
    print(loss)
    epochs = []
    for num, loss_ in enumerate(loss):

        epochs.append(
            {'epoch': num,
                'validation_loss': loss_}
        )

        stopping(epochs)
        print(num)
        print(stopping.stopping, "\n")
    print(epochs)

    import matplotlib.pyplot as plt

    losses = []
    for epoch in epochs:
        losses.append(epoch['validation_loss'])

    plt.plot(losses)
