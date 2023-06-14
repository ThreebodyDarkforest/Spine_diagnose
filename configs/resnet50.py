solver = dict(
    optim='Adam',
    lr_scheduler='Cosine',
    lr=1e-4,
)

saver = dict(
    save=True,
    save_every=5,
    save_best=True,
)