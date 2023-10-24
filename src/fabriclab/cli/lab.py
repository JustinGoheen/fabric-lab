import os
from pathlib import Path
from typing import List, Union

import typer
from rich import print as rprint
from typing_extensions import Annotated
from lightning_fabric.loggers.csv_logs import CSVLogger

from fabriclab import Config, LabModule, LabDataModule, LabTrainer

FILEPATH = Path(__file__)
PROJECTPATH = FILEPATH.parents[2]
PKGPATH = FILEPATH.parents[1]

app = typer.Typer()
docs_app = typer.Typer()
run_app = typer.Typer()
app.add_typer(docs_app, name="docs")
app.add_typer(run_app, name="run")


@app.callback()
def callback() -> None:
    pass


# DOCS APP


@docs_app.command("build")
def build_docs() -> None:
    import shutil

    os.system("mkdocs build")
    shutil.copyfile(src="README.md", dst="docs/index.md")


@docs_app.command("serve")
def serve_docs() -> None:
    os.system("mkdocs serve")


# RUN APP
@run_app.command("demo")
def run_demo():
    rprint("RUNNING DEMO")
    os.system(f"python {PKGPATH}/demo.py")


@run_app.command("demo-trainer")
def run_demo_trainer(
    devices: Annotated[int, typer.Option()] = 1,
    accelerator: Annotated[str, typer.Option()] = "auto",
    strategy: Annotated[str, typer.Option()] = "auto",
    num_nodes: Annotated[int, typer.Option()] = 1,
    precision: Annotated[str, typer.Option()] = "32-true",
) -> None:
    datamodule = LabDataModule()
    model = LabModule()
    trainer = LabTrainer(
        devices=devices,
        accelerator=accelerator,
        strategy=strategy,
        num_nodes=num_nodes,
        precision=precision,
        enable_checkpointing=True,
        max_epochs=2,
        logger=CSVLogger(save_dir=Config.CSVLOGGERPATH),
    )
    trainer.fit(model=model, datamodule=datamodule)


@run_app.command("streamlit")
def run_streamlit():
    os.system(f"streamlit run {PKGPATH / 'app/streamlit.py'}")
