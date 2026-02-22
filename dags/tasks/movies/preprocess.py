import json
import polars as pl
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer

DATA_BASE = Path("/home/gintoki/airflow/data")


def encoding(df, col_source, col_out):

    df = df.with_columns(
        pl.col(col_source)
        .map_elements(
            lambda x: [company["name"] for company in json.loads(x)],
            return_dtype=pl.List(pl.String),
        )
        .alias(col_out)
    )
    top = (
        df.select(col_out)
        .explode(col_out)
        .filter(pl.col(col_out).is_not_null())
        .group_by(col_out)
        .len()
        .sort("len", descending=True)
        .head(60)[col_out]
        .to_list()
    )
    lists = [
        [c for c in companies if c in set(top)] for companies in df[col_out].to_list()
    ]
    mlb = MultiLabelBinarizer(classes=top)
    encoded = mlb.fit_transform(lists)
    df = pl.DataFrame(encoded, schema=[f"{i}" for i in top]).cast(pl.Int8)

    return df


def clear_data():

    df = pl.read_csv(
        DATA_BASE / "raw/movie/tmdb_5000_movies.csv",
        schema_overrides={"runtime": pl.Float64},
    )
    print(df.shape)
    df = df.filter(
        pl.col("release_date").is_not_null()
        & pl.col("runtime").is_not_null()
        & pl.col("budget").is_not_null()
        & pl.col("revenue").is_not_null()
        & (pl.col("revenue") > 0)
        & (pl.col("budget") > 0)
    )
    print(df.shape)

    trash_columns = [
        "homepage",
        "tagline",
        "keywords",
        "status",
        "original_language",
        "spoken_languages",
        "id",
        "original_title",
        "overview",
        "title",
        "production_companies",
        "genres",
        "production_countries",
        "release_date",
        "popularity",
        "vote_average",
        "vote_count",
    ]

    df = (
        df.with_columns(pl.col("release_date").str.to_date().alias("date_parsed"))
        .with_columns(
            pl.col("date_parsed").dt.year().alias("release_year"),
            pl.col("date_parsed").dt.month().alias("month"),
        )
        .with_columns(
            (pl.col("month") * 2 * np.pi / 12).sin().alias("month_sin"),
            (pl.col("month") * 2 * np.pi / 12).cos().alias("month_cos"),
        )
        .drop(["date_parsed", "month"])
    )
    country_df = encoding(df, "production_countries", "coutries")
    genres_df = encoding(df, "genres", "genres")
    company_df = encoding(df, "production_companies", "companies")
    df = pl.concat([df, company_df], how="horizontal")
    df = pl.concat([df, genres_df], how="horizontal")
    df = pl.concat([df, country_df], how="horizontal")

    df = df.drop(trash_columns)

    df.write_csv(DATA_BASE / "processed/movie/movies.csv")

    return True
