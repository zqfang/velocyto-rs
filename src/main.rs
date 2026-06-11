use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "velocyto-rs",
    about = "RNA velocity analysis for single-cell RNA-seq data"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Run(velocyto::commands::run::RunArgs),
    Run10x(velocyto::commands::run10x::Run10xArgs),
    RunDropest(velocyto::commands::run_dropest::RunDropestArgs),
    RunSmartseq2(velocyto::commands::run_smartseq2::RunSmartseq2Args),
    DropestBcCorrect(velocyto::commands::dropest_bc_correct::DropestBcCorrectArgs),
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let verbose = match &cli.command {
        Commands::Run(a) => a.verbose,
        Commands::Run10x(a) => a.verbose,
        Commands::RunDropest(a) => a.verbose,
        Commands::RunSmartseq2(a) => a.verbose,
        Commands::DropestBcCorrect(a) => a.verbose,
    };
    let level = if verbose {
        log::LevelFilter::Debug
    } else {
        log::LevelFilter::Info
    };
    env_logger::builder().filter_level(level).init();
    match cli.command {
        Commands::Run(args) => velocyto::commands::run::run(args),
        Commands::Run10x(args) => velocyto::commands::run10x::run10x(args),
        Commands::RunDropest(args) => velocyto::commands::run_dropest::run_dropest(args),
        Commands::RunSmartseq2(args) => velocyto::commands::run_smartseq2::run_smartseq2(args),
        Commands::DropestBcCorrect(args) => {
            velocyto::commands::dropest_bc_correct::dropest_bc_correct(args)
        }
    }
}
