import marimo

__generated_with = "0.14.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import albumentations as A
    from PIL import Image

    from unets.data import CityscapesDataModule, overlay_mask
    return A, CityscapesDataModule, Image, overlay_mask


@app.cell
def _(A, CityscapesDataModule):
    dm = CityscapesDataModule(data_dir="../data/", batch_size=1, num_workers=1, pin_memory=False, transforms={"train": [A.ToTensorV2()], "eval": [A.ToTensorV2()]})
    return (dm,)


@app.cell
def _(dm):
    dm.setup("fit")
    train_dataloader = dm.train_dataloader()
    return (train_dataloader,)


@app.cell
def _(train_dataloader):
    sample = next(iter(train_dataloader))
    return (sample,)


@app.cell
def _(overlay_mask, sample):
    overlayed = overlay_mask(sample[0].squeeze(0).numpy(), sample[1].squeeze(0).numpy())
    return (overlayed,)


@app.cell
def _(Image, overlayed):
    Image.fromarray(overlayed)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
