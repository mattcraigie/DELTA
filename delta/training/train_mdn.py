
def train_mdn_model(train_loader, val_loader, num_epochs=100, hidden_dim=32, num_layers=4, k=10, compression_model=None):
    return None # mdn_model, {'train_losses': train_losses, 'val_losses': val_losses}


#
# compression_model = CompressionNetwork(reg_model)
#
# # Train the model
# model, history = train_model(
#     train_loader,
#     val_loader,
#     num_epochs=1000,
#     hidden_dim=hidden_dim + 1,
#     num_layers=1,
#     k=k,
#     compression_model=compression_model
# )