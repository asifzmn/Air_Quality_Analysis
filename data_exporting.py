import pandas as pd


def latex_simple_table_format(df):
    new_df = df.groupby([df.index.year]).agg(['min', 'mean', 'max'])
    new_df.index.set_names(['Year'], inplace=True)
    print(new_df.T.to_latex(col_space=3))


def latex_custom_table_format(stats):
    stats.to_csv('Files/general_stats.csv')
    # stats['count'] = stats['count'].astype('int')

    # stats = stats.iloc[:,[0,1,2,3,5,7]].round(1)

    latex_data = stats.to_latex(col_space=3).replace("\\\n", "\\ \hline\n").replace('\\toprule', '\\toprule\n\\hline')
    substring = latex_data[
                latex_data.index('\\begin{tabular}{') + len('\\begin{tabular}{') - 1:latex_data.index('}\n') + 1]
    latex_data = latex_data.replace(substring, '|'.join(substring))

    for axisName in stats.columns: latex_data = latex_data.replace(axisName, f"\\textbf{{{axisName}}}")
    # for axisName in stats.index: latex_data = latex_data.replace(axisName, f"\\textbf{{{axisName}}}")

    latex_data = latex_data.replace('25\%', "\\textbf{Q1}").replace('50\%', "\\textbf{Q2}").replace('75\%',
                                                                                                    "\\textbf{Q3}")
    latex_data = latex_data.replace('{Tungi}para', "{Tungipara}")

    print(latex_data)


def missing_data_fraction(timeseries):
    missing_percentage = (timeseries.isnull().sum() / len(timeseries) * 100).round(2)
    missing_percentage.to_csv('missing_percentage.csv')


def paper_comparison(comp_file):
    paper_data = pd.read_csv(comp_file, sep='\t')
    latex_custom_table_format(paper_data)
