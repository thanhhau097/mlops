import typer

from modeling.main import train_model, hyperparams_opt


app = typer.Typer()


@app.command()
def train(lr: 0.001, momentum: float = 0.9):
    typer.echo(f"Training with lr={lr} and momentum={momentum}")
    config = {
        "lr": lr,
        "momentum": momentum,
    }
    train_model(config)


@app.command()
def hyperopt():
    typer.echo("Running hyperparameter optimization")
    hyperparams_opt()
