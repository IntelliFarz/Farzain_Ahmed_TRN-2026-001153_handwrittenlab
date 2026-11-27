|   bug | faulty_code                                    | correction                                    | concept                      |
|------:|:-----------------------------------------------|:----------------------------------------------|:-----------------------------|
|     1 | StandardScaler().fit_transform(X) before split | Split first; fit scaler on training data only | Data leakage                 |
|     2 | Softmax(dim=1) before CrossEntropyLoss         | Return raw logits to CrossEntropyLoss         | Loss function expects logits |
|     3 | No opt.zero_grad() inside training loop        | Clear gradients before each backward pass     | Gradient accumulation        |
|     4 | Evaluate predictions on X_train                | Evaluate on held-out X_test                   | Train/test separation        |
|     5 | No model.eval() for inference with Dropout     | Call model.eval() before evaluation           | Training vs inference mode   |