solver = dict(
    optim='Adam',
    lr_scheduler='Cosine',
    lr=5e-5,
)

saver = dict(
    save=True,
    save_every=5,
    save_best=True,
)