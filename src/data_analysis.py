import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
import seaborn as sns


def describe_dataframe_core(df: DataFrame):
    describe_dataset(df)
    diagnostic_superclass_len(df)
    diagnostic_superclass(df)
    diagnostic_superclass_age(df)
    diagnostic_superclass_nurse(df)
    diagnostic_superclass_sex(df)
    diagnostic_superclass_device(df)
    correlation_heatmap(df)


def correlation_heatmap(df):
    _, ax = plt.subplots(figsize=(20, 20))
    colormap = sns.diverging_palette(220, 10, as_cmap=True)

    _ = sns.heatmap(
        df.corr(),
        cmap=colormap,
        square=True,
        cbar_kws={'shrink':.9},
        ax=ax,
        annot=True,
        linewidths=0.1, vmax=1.0, linecolor='white',
        annot_kws={'fontsize':8}
    )
    plt.title('Pearson Correlation of Features', y=1.05, size=15)


def diagnostic_superclass_len(df_main: DataFrame):
    df = df_main
    vc = df['diagnostic_superclass'].value_counts()
    sns.set_style("whitegrid")
    bar, ax = plt.subplots(figsize=(10, 6))
    ax = sns.barplot(x=vc.values/vc.values.sum()*100., y=vc.index, ci=None, palette="muted", orient='h')
    ax.set_title("Diagnostic Superclass Len Distribution", fontsize=20)
    ax.set_xlabel("percentage over all samples")
    ax.set_ylabel("diagnostic_superclass_len")
    for rect in ax.patches:
        ax.text(rect.get_width(), rect.get_y() + rect.get_height() / 2, "%.1f%%"% rect.get_width(), weight='bold')


def diagnostic_superclass(df_main: DataFrame):
    df = df_main
    vc = df['diagnostic_superclass'].value_counts()
    sns.set_style("whitegrid")
    bar, ax = plt.subplots(figsize=(10, 6))
    ax = sns.barplot(x=vc.values/df.shape[0]*100., y=vc.index, ci=None, palette="muted", orient='h')
    ax.set_title("diagnostic Superclass Distribution", fontsize=20)
    ax.set_xlabel("percentage over all samples")
    ax.set_ylabel("diagnostic superclass")
    for rect in ax.patches:
        ax.text(rect.get_width(), rect.get_y() + rect.get_height() / 2, "%.1f%%"% rect.get_width(), weight='bold')


def diagnostic_superclass_age(df_main: DataFrame):
    df = df_main
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title('Age Distributions of Different Superclass')

    for superclass in ['diagnostic_superclass']:
        data = df.loc[df[superclass] == 1]['age']
        sns.distplot(data, label=superclass)

    plt.legend(loc='upper left')
    plt.show()


def diagnostic_superclass_nurse(df_main: DataFrame, superclass_cols=['diagnostic_superclass']):
    df = df_main
    sns.set_style("whitegrid")
    bar, ax = plt.subplots(figsize=(10, 20))

    ax.set_title("diagnostic Superclass Distribution of Different Nurse", fontsize=20)

    all_index, all_count, all_values = [], [], []
    for nurse in df.nurse.unique():
        vc = df.loc[df.nurse == nurse][superclass_cols].sum(axis=0)
        all_index += list(vc.index)
        all_count += list(vc.values/df.shape[0]*100.)
        all_values += [nurse]*len(vc)

    df = pd.DataFrame()
    df['diagnostic superclass'] = all_index
    df['percentage over all samples'] = all_count
    df['nurse'] = all_values

    ax = sns.barplot(data=df, x="percentage over all samples", y="diagnostic superclass", hue="nurse", ci=None, orient='h')
    for rect in ax.patches:
        ax.text(rect.get_width(), rect.get_y() + rect.get_height() / 2,"%.1f%%"% rect.get_width(), weight='bold')


def diagnostic_superclass_sex(df_main: DataFrame, superclass_cols=['diagnostic_superclass']):
    df = df_main
    sns.set_style("whitegrid")
    bar, ax = plt.subplots(figsize=(10, 6))

    ax.set_title("Diagnostic Superclass Distribution of Different Sex", fontsize=20)

    all_index, all_count, all_values = [], [], []
    for sex in df.sex.unique():
        vc = df.loc[df.sex == sex][superclass_cols].sum(axis=0)
        all_index += list(vc.index)
        all_count += list(vc.values/df.shape[0]*100.)
        all_values += [sex]*len(vc)

    df = pd.DataFrame()
    df['diagnostic superclass'] = all_index
    df['percentage over all samples'] = all_count
    df['sex'] = all_values

    ax = sns.barplot(data=df, x="percentage over all samples", y="diagnostic superclass", hue="sex",ci=None, orient='h')
    for rect in ax.patches:
        ax.text(rect.get_width(), rect.get_y() + rect.get_height() / 2, "%.1f%%"% rect.get_width(), weight='bold')


def diagnostic_superclass_device(df_main: DataFrame, superclass_cols=['diagnostic_superclass']):
    df = df_main
    sns.set_style("whitegrid")
    bar,ax = plt.subplots(figsize=(10, 20))

    ax.set_title("diagnostic Superclass Distribution of Different Device", fontsize=20)

    all_index, all_count, all_values = [], [], []
    for device in df.device.unique():
        vc = df.loc[df.device == device][superclass_cols].sum(axis=0)
        all_index += list(vc.index)
        all_count += list(vc.values/df.shape[0]*100.)
        all_values += [device]*len(vc)

    df = pd.DataFrame()
    df['diagnostic superclass'] = all_index
    df['percentage over all samples'] = all_count
    df['device'] = all_values

    ax = sns.barplot(data=df, x="percentage over all samples", y="diagnostic superclass", hue="device",ci=None, orient='h')
    for rect in ax.patches:
        ax.text(rect.get_width(), rect.get_y() + rect.get_height() / 2,"%.2f%%"% rect.get_width(), weight='bold')


def describe_dataset(df: DataFrame):
    pd.set_option('display.max_columns', 30)
    print(df.describe())
