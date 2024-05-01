import pandas as pd
import numpy as np
from multiprocessing import Pool
from functools import partial
import re
import psutil
import functools
import datetime

end_datetime = datetime.datetime(2024, 5, 31, 23, 59, 59)

def time_limit(end_datetime):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if datetime.datetime.now() > end_datetime:
                raise Exception("The period of use has expired.")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@time_limit(end_datetime)
def safediv(x, y):
    
    if abs(y) > 0.0000001:
        return x/y
    return 0

@time_limit(end_datetime)
def replace_term(str_formula, term):
    # pattern = rf"(?<!\w){term}(?!\w)"
    pattern = rf"(?<=\b){term}(?=\b)"

    # pattern = rf"(?<=\b)(?<!\['|\")({term})(?!\b|'\]|\")"
    return re.sub(pattern, f"x['{term}']", str_formula)

# @time_limit(end_datetime)
# def transform_formula(formula, list_terms):
#     formula_for_eval = formula
#     for term in list_terms:
#         formula_for_eval = replace_term(formula_for_eval, term)
#     return formula_for_eval

@time_limit(end_datetime)
def transform_formula(formula, list_terms):
    elements = re.split('([*+\-/(), ])', formula)
    transformed = [f"x['{e.strip()}']" if e.strip() in list_terms else e for e in elements]
    return ''.join(transformed)


# @time_limit(end_datetime)
# def transform_formula_with_replaced_term(formula, list_terms, replaced_term):
#     elements = re.split('([*+\-/(), ])', formula)
#     transformed = [f"x['{e.strip()}']" if ((e.strip() in list_terms) and (e != replaced_term)) else f"x['rep_' + '{e.strip()}']" if (e.strip() in list_terms and (e == replaced_term)) else e for e in elements]
#     return ''.join(transformed)

# @time_limit(end_datetime)
# def transform_formula(formula):
#     elements = re.split('([*+\-/,\(\)])', formula)
#     transformed = [f"x['{e.strip()}']" if e.strip().isalnum() or '.' in e or ':' in e else e for e in elements]
#     return ''.join(transformed)

# def generate_kpi_series(str_kpi_formula, df_data):
#     return df_data.apply(lambda x: eval(transform_formula(str_kpi_formula, x.index)), axis=1)
@time_limit(end_datetime)
def extract_terms(input_str):
    return re.findall(r'[a-zA-Z0-9_.]+', input_str)

@time_limit(end_datetime)
def extract_unique_metric_terms_from_formula(input_str):
    list_terms =re.findall(r'[a-zA-Z0-9_.]+', input_str)
    if 'safediv' in list_terms:
        list_terms.remove('safediv')
    list_terms = sorted(list(set(list_terms)))
    return list_terms

# def extract_agg_metric_terms_from_formula(input_str):
#     list_terms =re.findall(r'(sum|mean|average|max|min|median)\([a-zA-Z0-9_]+\)', input_str)
#     return list_terms

@time_limit(end_datetime)
def generate_kpi_series(str_kpi_formula, df_data):
    list_terms = list(set(extract_terms(str_kpi_formula)))
    if 'safediv' in list_terms:
        list_terms.remove('safediv')
    # print('list_terms:', list_terms)
    
    return df_data.apply(lambda x: eval(transform_formula(str_kpi_formula, list_terms)), axis=1)
    # return df_data.apply(lambda x: eval(transform_formula(str_kpi_formula)), axis=1)




@time_limit(end_datetime)
def generate_kpi_series_multiprocess(str_kpi_formula, df_data, n_process = None):
    if n_process is None:
        n_process = max(1, int(psutil.cpu_count()*0.9))
    else:
        n_process = min(n_process, int(psutil.cpu_count()*0.9))

    df_split = [df_data.iloc[i::n_process] for i in range(n_process)]
    pool = Pool(n_process)
    partial_func = partial(generate_kpi_series, str_kpi_formula)
    df = pd.concat(pool.map(partial_func, df_split))
    pool.close()
    pool.join()

    return df
    




# @time_limit(end_datetime)
# def generate_whatif_kpi_series_with_replaced_term(str_kpi_formula, replaced_term, df_data):
#     list_terms = list(set(extract_terms(str_kpi_formula)))
#     if 'safediv' in list_terms:
#         list_terms.remove('safediv')
#     # print('list_terms:', list_terms)
#     # list_terms_new = [f"mean_{term}" if replaced_term == term else term for term in list_terms]
    
#     return df_data.apply(lambda x: eval(transform_formula_with_replaced_term(str_kpi_formula, list_terms, replaced_term)), axis=1)



# @time_limit(end_datetime)
# def generate_whatif_kpi_series_with_replaced_term_multiprocess(str_kpi_formula, df_data, replaced_term, n_process = None):
#     if n_process is None:
#         n_process = max(1, int(psutil.cpu_count()*0.9))
#     else:
#         n_process = min(n_process, int(psutil.cpu_count()*0.9))

#     df_split = [df_data.iloc[i::n_process] for i in range(n_process)]
#     pool = Pool(n_process)
#     partial_func = partial(generate_whatif_kpi_series_with_replaced_term, str_kpi_formula, replaced_term)
#     df = pd.concat(pool.map(partial_func, df_split))
#     pool.close()
#     pool.join()

#     return df




@time_limit(end_datetime)
def generate_contribution_series(kpi_name, list_terms, df_data_and_kpi,):
    df_contribution = pd.DataFrame()

    for term in list_terms:
        df_contribution[term] = df_data_and_kpi[term]/df_data_and_kpi[kpi_name]
    
    return df_contribution

@time_limit(end_datetime)
def generate_contribution_series_multiprocess(kpi_name, list_terms, df_data_and_kpi, n_process = None):
    if n_process is None:
        n_process = max(1, int(psutil.cpu_count()*0.9))
    else:
        n_process = min(n_process, int(psutil.cpu_count()*0.9))

    df_split = [df_data_and_kpi.iloc[i::n_process] for i in range(n_process)]
    pool = Pool(n_process)
    partial_func = partial(generate_contribution_series, kpi_name, list_terms)
    df = pd.concat(pool.map(partial_func, df_split))
    pool.close()
    pool.join()

    return df

# @time_limit(end_datetime)
# def get_highest_corr_ts(df, target_mertic, list_candidate_mertic, absolute=False, corr_method='pearson'):
#     ds_corr = df[[target_mertic]+list_candidate_mertic].corr(method=corr_method)[target_mertic].drop(target_mertic, axis=0)
#     if absolute:
#         arg_max_index = ds_corr.abs().argmax()
#     else:
#         arg_max_index = ds_corr.argmax()

#     return ds_corr.index[arg_max_index], ds_corr.iloc[arg_max_index]


@time_limit(end_datetime)
def get_corr_for_target_metric_sorted(df, target_mertic, list_candidate_mertic, sort_type='absolute', corr_method='pearson'):

    ds_corr = df[[target_mertic]+list_candidate_mertic].corr(method=corr_method)[target_mertic].drop(target_mertic, axis=0)
    if sort_type == 'absolute':
        return ds_corr.sort_values(ascending=False, key=abs)
        
    elif sort_type == 'positive':
        return ds_corr.sort_values(ascending=False)
    elif sort_type == 'negative':
        return ds_corr.sort_values(ascending=True)
    else:
        raise Exception("Wrong argument value for type.")


# @time_limit(end_datetime)
# def indentify_correlated_metric_term_with_kpi(df, kpi_name, dict_kpi_formula):
#     list_metric_terms = extract_unique_metric_terms_from_formula(dict_kpi_formula[kpi_name]["formula"])
#     metric_term, corr_coeff = get_highest_corr_ts(df[[kpi_name] + list_metric_terms], kpi_name, list_metric_terms, absolute=True)

#     return metric_term, corr_coeff

@time_limit(end_datetime)
def indentify_correlated_metric_term_with_kpi(df, kpi_name, dict_kpi_formula):
    list_metric_terms = extract_unique_metric_terms_from_formula(dict_kpi_formula[kpi_name]["formula"])
    # metric_term, corr_coeff = get_highest_corr_ts(df[[kpi_name] + list_metric_terms], kpi_name, list_metric_terms, absolute=True)

    ds_corr = get_corr_for_target_metric_sorted(df[[kpi_name] + list_metric_terms], kpi_name, list_metric_terms, sort_type='absolute', corr_method='pearson')
    metric_term, corr_coeff = ds_corr.index[0], ds_corr.iloc[0]
        # metric_term, corr_coeff = get_highest_corr_ts(df[[kpi_name] + list_metric_terms], kpi_name, list_metric_terms, absolute=True)

    return metric_term, corr_coeff


@time_limit(end_datetime)
def get_kpi_corr_of_terms(df, kpi_name, dict_kpi_formula):
    list_metric_terms = extract_unique_metric_terms_from_formula(dict_kpi_formula[kpi_name]["formula"])
    # metric_term, corr_coeff = get_highest_corr_ts(df[[kpi_name] + list_metric_terms], kpi_name, list_metric_terms, absolute=True)
    ds_corr = get_corr_for_target_metric_sorted(df[[kpi_name] + list_metric_terms], kpi_name, list_metric_terms, sort_type='absolute', corr_method='pearson')

    return ds_corr

@time_limit(end_datetime)
def extract_degraded_samples(timeseries, th_percentile, degradation_direction='postive'):
    if degradation_direction == 'postive':
        degraded_samples = timeseries[timeseries >= np.percentile(timeseries, th_percentile)]
    elif degradation_direction == 'negative':
        degraded_samples = timeseries[timeseries <= np.percentile(timeseries, th_percentile)]
    else:
        print('Wrong degradation direction setting')
        return None

    return degraded_samples.index, degraded_samples.values

@time_limit(end_datetime)
def parse_agg_formula(formula, input_agg_term_prefix='agg_' , remove_str = 'safediv(', add_prefix_agg = True):

    formula_new = formula.replace(remove_str, '')
    if input_agg_term_prefix == 'agg_':
        # parsed_func_var_pairs = re.findall(r'\b(agg_max|agg_min|agg_sum|agg_mean|agg_median)\b\(([a-zA-Z0-9_]+)\)', formula_new)
        parsed_func_var_pairs = re.findall(r'\b(agg_max|agg_min|agg_sum|agg_mean|agg_median)\b\s*\(\s*([a-zA-Z0-9_.]+)\s*\)', formula_new)
        
        func_var_pairs = [(func.replace('agg_', ''), var) for func, var in parsed_func_var_pairs]
    else:
        parsed_func_var_pairs = re.findall(r'\b(max.|min.|sum.|mean.|median.)\b([a-zA-Z0-9_]+)', formula_new)
        func_var_pairs = [(func.replace('.', ''), var) for func, var in parsed_func_var_pairs]
    # print('parsed_func_var_pairs:', parsed_func_var_pairs)
    # print('func_var_pairs:', func_var_pairs)

    # dict for mapping of functions to variables
    # dict_term_agg = {var: func.replace('agg_', '') for func, var in func_var_pairs}
    if add_prefix_agg == True:
        if input_agg_term_prefix == 'agg_':
            dict_term_agg = {f"{func}({var})": (var, func) for func, var in func_var_pairs}
            # dict_term_agg = {func :(func.replace('.', ''), var) for func, var in parsed_func_var_pairs}
        else:
            dict_term_agg = {func +'.'+ var: (var, func) for func, var in func_var_pairs}
    else:
        dict_term_agg = {var: (var, func) for func, var in func_var_pairs}

    return dict_term_agg

@time_limit(end_datetime)
def generate_agg_terms_for_formula(df_data, formula, agg_period, agg_dim_col, timestamp_col = 'DateTime',
                                    input_agg_term_prefix='agg_', add_prefix_agg=True):

    if agg_period in ['H', 'Hour']:
        period_col = 'Datetime'
        period = 'h'
    if agg_period in ['D', 'Day']:
        period_col = 'Date'
        period = 'D'
    if agg_period in ['M', 'Month']:
        period_col = 'Year-Month'
        period = 'M'
    if agg_period in ['Y', 'Year']:
        period_col = 'Year'
        period = 'Y'
    
    df_data[period_col] = df_data[timestamp_col].dt.to_period(period)
    dict_term_agg = parse_agg_formula(formula, input_agg_term_prefix=input_agg_term_prefix, add_prefix_agg=add_prefix_agg)

    if agg_period == None:
        df_agg_terms = df_data.groupby([agg_dim_col]).agg(**dict_term_agg)
    elif agg_dim_col == None:
        df_agg_terms = df_data.groupby([period_col]).agg(**dict_term_agg)
    else:
        df_agg_terms = df_data.groupby([agg_dim_col, period_col]).agg(**dict_term_agg)

    return df_agg_terms


@time_limit(end_datetime)
def extract_common_agg_terms(dict_kpi_agg_formula, input_agg_term_prefix=None, add_prefix_agg=True):
    dict_term_agg = dict()
    # list_kpi = dict_kpi_agg_formula.keys()
    for kpi_name, item in dict_kpi_agg_formula.items():
        formula = item['formula']
        # print(f'kpi name: {kpi_name},  formula: {formula}')
        tmp_dict_term_agg = parse_agg_formula(formula, input_agg_term_prefix=input_agg_term_prefix, add_prefix_agg=add_prefix_agg)
        dict_term_agg = dict_term_agg | tmp_dict_term_agg
        # print(f'tmp_dict_term_agg: {tmp_dict_term_agg},  dict_term_agg: {dict_term_agg}')
    return dict_term_agg



@time_limit(end_datetime)
def generate_agg_metric(df_data, dict_term_agg, agg_period, agg_dim_col, timestamp_col = 'DateTime'):

    if agg_period in ['H', 'Hour']:
        period_col = 'Datetime'
        period = 'h'
    if agg_period in ['D', 'Day']:
        period_col = 'Date'
        period = 'D'
    if agg_period in ['M', 'Month']:
        period_col = 'Year-Month'
        period = 'M'
    if agg_period in ['Y', 'Year']:
        period_col = 'Year'
        period = 'Y'
    
    df_data[period_col] = df_data[timestamp_col].dt.to_period(period)

    if agg_period == None:
        df_agg_metric = df_data.groupby([agg_dim_col]).agg(**dict_term_agg)
    elif agg_dim_col == None:
        df_agg_metric = df_data.groupby([period_col]).agg(**dict_term_agg)
    else:
        df_agg_metric = df_data.groupby([agg_dim_col, period_col]).agg(**dict_term_agg)

    return df_agg_metric


@time_limit(end_datetime)
def generate_agg_metric_for_all_kpis(df_data, dict_kpi_agg_formula, agg_period, agg_dim_col, timestamp_col = 'DateTime',
                                    input_agg_term_prefix='agg_', add_prefix_agg=True):
    
    dict_term_agg = extract_common_agg_terms(dict_kpi_agg_formula, input_agg_term_prefix=input_agg_term_prefix, add_prefix_agg=add_prefix_agg)
    df_agg_metric = generate_agg_metric(df_data, dict_term_agg, agg_period, agg_dim_col, timestamp_col = timestamp_col)

    return df_agg_metric



@time_limit(end_datetime)
def generate_agg_kpi_with_agg_formula(df_terms, kpi_name, agg_formula, agg_period='D', agg_dim_col= None, 
                                      timestamp_col='DateTime', input_agg_term_prefix='agg_', add_prefix_agg=False,  n_process=None):
    
    try:
        df_agg_terms = generate_agg_terms_for_formula(df_terms, agg_formula, agg_period=agg_period, 
                                                    agg_dim_col= agg_dim_col, timestamp_col=timestamp_col,
                                                    input_agg_term_prefix=input_agg_term_prefix,
                                                    add_prefix_agg=add_prefix_agg
                                                    )
    except Exception as e:
        raise Exception("Error occurred: ", e)
        

    if input_agg_term_prefix=='agg_':
        def transform_agg_formula(before_formula):
            after_formula = re.sub(r'agg_([a-z]+)\((\w+)\)', r'\1.\2', before_formula)

            return after_formula

        new_agg_formula = transform_agg_formula(agg_formula)
    else:
        new_agg_formula = agg_formula

    df_agg_kpi = generate_kpi_series_multiprocess(new_agg_formula, df_agg_terms, n_process)
    df_agg_kpi.name = kpi_name

    return df_agg_kpi

@time_limit(end_datetime)
def generate_whatif_kpi(target_kpi, dict_kpi_formula, df_metric, replaced_metric, df_metric_stats, list_index_metric_stats, list_keep_dim_col=[], replace_stats_type='mean'):

    ### Usasge example:
    # df_kpi_whatif = generate_whatif_kpi(
    #     target_kpi=target_kpi, 
    #     dict_kpi_formula=dict_kpi_formula, 
    #     df_metric=df_ts_generation, 
    #     replaced_metric='generation.fossil.gas', 
    #     df_metric_stats=df_main_stats_generation_by_country_month, 
    #     list_index_metric_stats=['AreaName', 'Year-Month'], 
    #     list_keep_dim_col=['AreaName', 'DateTime'],
    #     replace_stats_type='mean')


    if  dict_kpi_formula[target_kpi]['type'] == 'ratio_using_num_den':
        raise ValueError("ration_using_num_dem KPI not supported")
        return
    # if dict_kpi_formula[target_kpi]['type'] != 'ratio_using_num_den':
        # df_metric_whatif = df_metric.copy()
        # df_metric_whatif[replaced_term] = df_metric_whatif.apply(lambda x: df_main_stats_generation_by_country_month.loc[x['AreaName']].iloc[0][replaced_term]['mean'], axis=1)
    if len(list_index_metric_stats) == 1:
        ds_metric_whatif = df_metric.apply(lambda x: df_metric_stats.loc[x[list_index_metric_stats[0]]][replaced_metric][replace_stats_type], axis=1)
    elif len(list_index_metric_stats) == 2:
        ds_metric_whatif = df_metric.apply(lambda x: df_metric_stats.loc[x[list_index_metric_stats[0]]].loc[x[list_index_metric_stats[1]]][replaced_metric][replace_stats_type], axis=1)
    else:
        raise ValueError("1 or 2 indexes are suppored for metric stats")
    
    if len(list_keep_dim_col) == 0:
        df_kpi_whatif = pd.DataFrame(index=df_metric.index)
    else:
        df_kpi_whatif = df_metric[list_keep_dim_col]

    df_metric_whatif = df_metric.drop(columns=[replaced_metric])
    df_metric_whatif[replaced_metric] = ds_metric_whatif
    
    item = dict_kpi_formula[target_kpi]
    kpi_type = item['type']
    str_formula = item['formula']
    print(f'{target_kpi} := {str_formula}')

    if kpi_type == 'normal': 
        df_tmp = pd.DataFrame({target_kpi: generate_kpi_series_multiprocess(item['formula'], df_metric_whatif)})
        df_kpi_whatif = pd.concat([df_kpi_whatif, df_tmp], axis=1)

    elif kpi_type == 'ratio':
        df_tmp = pd.DataFrame({target_kpi: generate_kpi_series_multiprocess(item['formula'], df_metric_whatif)})
        df_kpi_whatif = pd.concat([df_kpi_whatif, df_tmp], axis=1)

    else:
        raise ValueError("Not supported KPI formula type.  Only 'normal' or 'ratio' type is supported")
            
    return df_kpi_whatif



@time_limit(end_datetime)
def generate_metric_whatif_contribution(df_kpi, df_metric, df_metric_stats, dict_kpi_formula, target_kpi, list_contributing_metric, list_index_metric_stats, replace_stats_type='mean'):

    ### Usasge example:
    # df_metric_whatif_contribution = generate_metric_whatif_contribution(
    #     df_kpi=df_kpi, 
    #     df_metric=df_ts_generation, 
    #     df_metric_stats=df_main_stats_generation_by_country_month, 
    #     dict_kpi_formula=dict_kpi_formula, 
    #     target_kpi=target_kpi, 
    #     list_contributing_metric=extract_unique_metric_terms_from_formula(dict_kpi_formula[target_kpi]['formula']), 
    #     list_index_metric_stats=['AreaName', 'Year-Month'], 
    #     replace_stats_type='mean')    

    df_metric_whatif_contribution = pd.DataFrame(index=df_metric.index, columns=df_metric.columns)
    
    for metric in list_contributing_metric:
        print(f"checking {metric}..")
        df_kpi_whatif = generate_whatif_kpi(
            target_kpi=target_kpi,
            dict_kpi_formula=dict_kpi_formula,
            df_metric=df_metric, 
            replaced_metric=metric, 
            df_metric_stats=df_metric_stats, 
            list_index_metric_stats=list_index_metric_stats, 
            list_keep_dim_col=[], 
            replace_stats_type=replace_stats_type)
    
        df_metric_whatif_contribution[metric] = (df_kpi_whatif[target_kpi] - df_kpi[target_kpi])/df_kpi[target_kpi]

    return df_metric_whatif_contribution
        
