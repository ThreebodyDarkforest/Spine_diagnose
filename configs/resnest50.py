solver = dict(
    optim='Adam',
    lr_scheduler='cos',
    lr=2e-5,
    last_gamma=False,
    weight_decay=0.,
    momentum=0.,
    warmup_epochs=0,
)

saver = dict(
    save=True,
    save_every=5,
    save_best=True,
)

model = dict(
    final_drop=0,
)