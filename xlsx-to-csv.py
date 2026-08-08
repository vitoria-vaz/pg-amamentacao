import argparse
from pathlib import Path

import pandas as pd


def convert_xlsx_to_csv(xlsx_path: str, csv_path: str, sheet_name=0) -> None:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description="Converte .xlsx para .csv")
    parser.add_argument("xlsx", help="Caminho do arquivo .xlsx")
    parser.add_argument(
        "-o",
        "--output",
        help="Caminho do arquivo .csv de saída",
        default=None,
    )
    parser.add_argument(
        "-s",
        "--sheet",
        help="Nome ou índice da planilha. Padrão: primeira planilha",
        default=0,
    )

    args = parser.parse_args()

    xlsx_file = Path(args.xlsx)
    if not xlsx_file.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {xlsx_file}")

    output_file = Path(args.output) if args.output else xlsx_file.with_suffix(".csv")
    convert_xlsx_to_csv(str(xlsx_file), str(output_file), sheet_name=args.sheet)

    print(f"Convertido com sucesso: {output_file}")


if __name__ == "__main__":
    main()