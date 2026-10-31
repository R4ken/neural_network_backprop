from common.data_loader import load_mnist_data
from py_nn1 import (
    MLP,
    ReLU,
    Softmax,
    CategoricalCrossEntropyLoss,
    SGD,
    compute_multiclass_accuracy,
    set_seed
)
import numpy as np

set_seed(54217)

data = load_mnist_data()
rng = np.random.default_rng(54217)
indices = rng.permutation(len(data.x_train))

eval_indices = indices[:5000]
train_indices = indices[5000:]

x_eval = data.x_train[eval_indices]
y_eval = data.y_train[eval_indices]

x_train = data.x_train[train_indices]
y_train = data.y_train[train_indices]

model = MLP.from_dimensions (
    layer_dims=[784, 150, 50, 10],
    hidden_activation=ReLU,
    output_activation=Softmax,
    use_bias=True,
    weight_init="he"
)

loss_fn = CategoricalCrossEntropyLoss()
optimizer = SGD(learning_rate=0.05)

history = model.fit(
    x_train=x_train,
    y_train=y_train,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=30,
    batch_size=32,
    validation_data=(x_eval, y_eval),
    metrics={"accuracy": compute_multiclass_accuracy},
    verbose=True
)

test_loss = model.evaluate(data.x_test, data.y_test, loss_fn)
print(f"Loss: {test_loss:.5f}")

probabilities = model.predict(data.x_test)
test_accuracy = compute_multiclass_accuracy(
    probabilities,
    data.y_test
)

print(f"Accuracy: {test_accuracy:.3f}")