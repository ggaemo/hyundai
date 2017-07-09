import pandas as pd
import re
import numpy as np
import os
import collections
import multiprocessing as mp


data_dir = '/home/jinwon/PycharmProjects/hyundai/data'
result_dir = '/home/jinwon/PycharmProjects/hyundai/simple_search/result'
def filter(df, column, value):
    return df.loc[df[column] == value]

def clean_search_list(query_name, name_list):
    result = False
    if query_name == 'nan':
        return result
    query_name = re.sub('\(.*\)', '', query_name)
    query_name = re.sub('^\s+|\s+$', '', query_name)
    query_name = re.sub('[\(\(]', '', query_name)  # 시트앗세이-운전석(멀티펑션 : 이런애가 있음..
    for name in name_list:
        name = re.sub('\(.*\)', '', name)
        name = re.sub('^\s+|\s+$', '', name)
        name = re.sub('[\(\(]', '', name) # 시트앗세이-운전석(멀티펑션 : 이런애가 있음..
        try:
            if re.match(query_name, name):
                result = True
                break
        except:
            break
    return result

def clean_partial_search_list(query_name, name_list, n_gram, threshold):
    result = False
    if query_name == 'nan':
        return result
    query_name = re.sub('\(.*\)', '', query_name)
    query_name = re.sub('^\s+|\s+$', '', query_name)
    query_name = re.sub('[\(\(]', '', query_name) # 시트앗세이-운전석(멀티펑션 : 이런애가 있음..
    for name in name_list:
        name = re.sub('\(.*\)', '', name)
        name = re.sub('^\s+|\s+$', '', name)
        name = re.sub('[\(\(]', '', name) # 시트앗세이-운전석(멀티펑션 : 이런애가 있음..
        cnt = 0
        if len(name) - n_gram > 0 and abs(len(name) - len(query_name)) < 5:
            for i in range(len(name) - n_gram):
                partial_name = name[i:i+n_gram]
                if partial_name in query_name:
                    cnt += 1
            if cnt > (len(name) - n_gram) * threshold:
                print('partial match', query_name, '|', name)
                result = True
    return result

def get_item_no(val_1, val_2):
    pattern = re.match('([A-Za-z\d]{10,11})', val_1)
    pattern_desc = re.search('([A-Za-z\d]{5})[- ]?([A-Za-z\d]{5,6})', val_2)
    if pattern:
        item_no = pattern.group(1)
    elif pattern_desc:
        group_2 = re.sub('[- ]', '', pattern_desc.group(2))
        item_no = pattern_desc.group(1) + group_2
    else:
        item_no = None
    return item_no

# bpts = pd.read_csv(os.path.join(data_dir, 'CRDN_EPWSMCDGBPTS.csv'), low_memory=False)
bptm = pd.read_csv(os.path.join(data_dir, 'CRDN_EPWSMCDGBPTM.csv'), low_memory=False)
mendi = pd.read_csv(os.path.join(data_dir, 'CRDN_EPWSMCDMENDI.csv'), low_memory=False)
mtd = pd.read_csv(os.path.join(data_dir, 'CRDN_EPWSMCDMCMTD.csv'), low_memory = False)
mei = pd.read_csv(os.path.join(data_dir, 'CRDN_EPWSMCDMCMEI.csv'), low_memory = False)

mei['ITEM_NO'].fillna('nan', inplace=True)
mei['ITEM_NM_EN'].fillna('nan', inplace=True)
mei['ITEM_NM_KO'].fillna('nan', inplace=True)
mei['APLY_DESC'].fillna('nan', inplace=True)
mendi['ITEM_NO'].fillna('nan', inplace=True)

mei['ITEM_NO'] = mei['ITEM_NO'].apply(str)
mei['APLY_DESC'] = mei['APLY_DESC'].apply(str)
mendi['ITEM_NO'] = mendi['ITEM_NO'].apply(str)
bptm['PART_NO'] = bptm['PART_NO'].apply(str)


mendi['ITEM_NO'] = mendi['ITEM_NO'].apply(func = lambda x: re.sub('[\s-]', '', x))
bptm['PART_NO'] = bptm['PART_NO'].apply(func = lambda x: re.sub('[\s-]', '', x))
mei['ITEM_NO'] = mei['ITEM_NO'].apply(func = lambda x: re.sub('[\s-]', '', x))

hw_name_en = pd.read_csv(os.path.join(data_dir, '현재_하드웨어_배정_영어.csv'))
hw_name_ko = pd.read_csv(os.path.join(data_dir, '현재_하드웨어_배정_한글.csv'))
hw_name  = pd.concat([hw_name_ko, hw_name_en])

hw_bptm_unmatched_name_ko = pd.read_csv(os.path.join(data_dir,
                                                     'bptm_unmatched_item_no기준_한글_명칭들.csv'))
hw_bptm_unmatched_name_en = pd.read_csv(os.path.join(data_dir,
                                                     'bptm_unmatched_item_no기준_영어_명칭들.csv'))
hw_bptm_unmatched_name = pd.concat([hw_bptm_unmatched_name_ko,
                                    hw_bptm_unmatched_name_en])

# bptm_unmatched_hw_item_no = pd.read_csv(os.path.join(data_dir, 'not_matched_hw.csv'))

hw_name = pd.concat([hw_name, hw_bptm_unmatched_name]).drop_duplicates()['NAME'].values

bptm_hw_name = pd.read_csv(os.path.join(data_dir, 'bptm_hw_name.csv'))['NAME'].values

mupg = pd.read_csv(os.path.join(data_dir, 'CRDN_EPWSMCDGMUPG.csv'), low_memory= False)
mupg['PROJ_CCAR'] = mupg['PROJ_CCAR'].apply(lambda x: re.sub('\s', '', x))
mupg['UPGO_N'] = mupg['UPGO_N'].apply(lambda x: re.sub('\s', '', x))
mupg = mupg[['PROJ_CCAR', 'PROJ_CYY', 'UPGO_N', 'MDFY_DTM']]
mupg = mupg.sort_values(['PROJ_CCAR', 'PROJ_CYY', 'UPGO_N', 'MDFY_DTM'])
mupg = mupg.drop_duplicates(subset=['PROJ_CCAR', 'PROJ_CYY', 'UPGO_N'], keep = 'last') #

mei_mupg = pd.merge(mei[['MTD_ID','ITEM_NO','APLY_DESC' ,'ITEM_NM_EN','ITEM_NM_KO','QTY']],
                   mtd[['MTD_ID', 'UPG_NO']])

def get_upg_no_by_car_year(mupg, remove_trailing_U):
    if remove_trailing_U:
        print('trailing U car type remove')
    car_year = mupg[['PROJ_CCAR', 'PROJ_CYY']].drop_duplicates()
    upg_no_by_car = dict()
    for idx, (car, year) in car_year.iterrows():
        if remove_trailing_U:
            if car.endswith('U') and len(car) > 2:
                continue

        upg_no_by_car[(car, year)] = list()
        car_cond = mupg['PROJ_CCAR'] == car
        year_cond = mupg['PROJ_CYY'] == year
        upg_no_data = mupg.loc[car_cond & year_cond, 'UPGO_N']

        assert sum(upg_no_data.duplicated()) == 0, 'upg_no_duplicated'

        for idx, upg_no in enumerate(upg_no_data):
            match = re.search('([A-Za-z\d]{2})UPG([A-Za-z\d]{5,6})', upg_no)
            assert match, 'mupg upgo_n not matching pattern, {}'.format(upg_no)
            assert match.group(1) == car, '{}_{}_{}'.format(match.group(1), car, upg_no)
            assert not  match.group(2) in upg_no_by_car[(car, year)], \
                '{}_{}'.format(upg_no, match.group(2))
            upg_no_by_car[(car, year)].append(match.group(2))

    return upg_no_by_car

def eitem_hw_split_mei(mei, mtd_id_list, n_gram, threshold):
    end_item_list = list()
    rows = list()
    for idx, mtd_id in enumerate(mtd_id_list):
        skip_row = False
        if idx % 1000 == 0:
            print(idx / len(mtd_id_list))
        subset = filter(mei, 'MTD_ID', mtd_id)
        subset = subset[['ITEM_NO', 'APLY_DESC','ITEM_NM_KO', 'ITEM_NM_EN']]

        mei_part_split_row = dict()
        mei_part_split_row['E_LIST'] = list()
        mei_part_split_row['HW_LIST']  = list()
        for idx, row in subset.iterrows():
            item_no = get_item_no(row['ITEM_NO'], row['APLY_DESC'])
            if item_no:
                ko_nm = row['ITEM_NM_KO']
                en_nm = row['ITEM_NM_EN']
                bptm_search = filter(bptm, 'PART_NO', item_no)
                if not bptm_search.empty:
                    item_nm = bptm_search['PART_DESC'].values[0]
                    if item_nm in bptm_hw_name:
                        mei_part_split_row['HW_LIST'].append(item_no)
                    elif clean_search_list(item_nm, bptm_hw_name):
                        mei_part_split_row['HW_LIST'].append(item_no)
                    elif clean_partial_search_list(item_nm, bptm_hw_name, n_gram, threshold):
                        mei_part_split_row['HW_LIST'].append(item_no)
                    else:
                        mei_part_split_row['E_LIST'].append(item_no)
                        end_item_list.append((item_no, ko_nm, en_nm))
                else:
                    if clean_search_list(ko_nm, hw_name) or clean_search_list(en_nm,hw_name):
                        mei_part_split_row['HW_LIST'].append(item_no)
                    elif clean_partial_search_list(ko_nm, hw_name, n_gram, threshold) \
                            or clean_partial_search_list(en_nm,hw_name, n_gram, threshold):
                        mei_part_split_row['HW_LIST'].append(item_no)
                    else:
                        mei_part_split_row['E_LIST'].append(item_no)
                        end_item_list.append((item_no, ko_nm, en_nm))
            else:
                skip_row = True
                # break

        if not skip_row:
            mei_part_split_row['MTD_ID'] = mtd_id
            rows.append(mei_part_split_row)
    return rows, end_item_list

def get_car_year_one_eitem_mei(mtd_upg, car, year, upg_list, mei_partitioned):

    mtd_id_subset = mtd_upg.loc[mtd_upg['UPG_NO'].isin(upg_list)]
    mei_subset = pd.merge(mei_partitioned, mtd_id_subset, how = 'inner', on='MTD_ID')

    new_rows = list()
    for idx, row in mei_subset.iterrows():
        if idx % 1000 == 0:
            print(idx / len(mei_subset), car, year)
        # upg_no = row['UPG_NO']
        e_list = row['E_LIST']
        row.drop('E_LIST', inplace=True)
        if len(e_list) > 1:
            mtd_id = row['MTD_ID']
            for eitem_idx, eitem in enumerate(e_list):
                row = row.copy(deep=True)
                new_mtd_id = str(int(mtd_id)) + '_' + str(eitem_idx)
                row['E_ITEM'] = eitem
                row['MTD_ID'] = new_mtd_id
                new_rows.append(row)
        elif len(e_list) == 1:
            row['E_ITEM'] = e_list.pop()
            new_rows.append(row)
        else:
            row['E_ITEM'] = list()
            new_rows.append(row)
    output_split_by_eitem = pd.DataFrame(new_rows)
    output_split_by_eitem['CAR'] = car
    output_split_by_eitem['YEAR'] = year
    output_split_by_eitem = output_split_by_eitem[['CAR', 'YEAR', 'UPG_NO','MTD_ID', 'E_ITEM',
                                                   'HW_LIST']]
    output_split_by_eitem.to_pickle(os.path.join(result_dir, 'output_{}_{}.df'.format(car,
                                                                                   year)))
    return output_split_by_eitem

def get_output_data():

    if os.path.exists(os.path.join(result_dir, 'mei_partitioned.df')):
        print('retrieved mei_partitioned.df')
        mei_partitioned = pd.read_pickle(os.path.join(result_dir, 'mei_partitioned.df'))
    else:
        pool = mp.Pool()
        mtd_id_list = mei['MTD_ID'].unique()
        mtd_id_list_split = np.split(mtd_id_list, np.arange(mtd_id_list.size // 12, mtd_id_list.size, mtd_id_list.size // 12))

        result = [pool.apply_async(eitem_hw_split_mei, args = (mei, split, 4, 0.9)) for split in
                  mtd_id_list_split]
        retrieved = [p.get() for p in result]
        all_rows = list()
        end_item_list = list()
        for rows in retrieved:
            data_row, eitem_elem = rows
            end_item_list.extend(eitem_elem)
            all_rows.extend(data_row)

        mei_partitioned = pd.DataFrame(all_rows)

        mei_partitioned.to_pickle(os.path.join(result_dir, 'mei_partitioned.df'))

        pd.DataFrame(end_item_list).to_csv(os.path.join(result_dir, 'end_item_list.csv'))


    upg_no_by_car = get_upg_no_by_car_year(mupg, True)

    mtd_upg = mtd[['MTD_ID', 'UPG_NO']]

    pool = mp.Pool()
    result = list()
    for (car, year), upg_list in upg_no_by_car.items():
        result.append(pool.apply_async(get_car_year_one_eitem_mei,
                                       args = (mtd_upg, car, year, upg_list,
                                               mei_partitioned)))

    retrieved = [p.get() for p in result]



if __name__ =='__main__':
    get_output_data()