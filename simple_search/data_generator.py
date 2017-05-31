import multiprocessing as  mp
import os
import re

import numpy as np
import pandas as pd

data_dir = '/home/jinwon/PycharmProjects/hyundai/data'

mei = pd.read_csv(os.path.join(data_dir, 'CRDN_EPWSMCDMCMEI.csv'), low_memory = False)

mei['ITEM_NO']  = mei['ITEM_NO'].apply(str)
mei['ITEM_NO'] = mei['ITEM_NO'].apply(lambda x: re.sub('[- ]','',x))

mtd = pd.read_csv(os.path.join(data_dir, 'CRDN_EPWSMCDMCMTD.csv'), low_memory = False)

mupg = pd.read_csv(os.path.join(data_dir, 'CRDN_EPWSMCDGMUPG.csv'), low_memory= False)
mupg['PROJ_CCAR'] = mupg['PROJ_CCAR'].apply(lambda x: re.sub('\s', '', x))
mupg['UPGO_N'] = mupg['UPGO_N'].apply(lambda x: re.sub('\s', '', x))
mupg = mupg[['PROJ_CCAR', 'PROJ_CYY', 'UPGO_N', 'MDFY_DTM']]
mupg = mupg.sort_values(['PROJ_CCAR', 'PROJ_CYY', 'UPGO_N', 'MDFY_DTM'])
mupg = mupg.drop_duplicates(subset=['PROJ_CCAR', 'PROJ_CYY', 'UPGO_N'], keep = 'last') #
#  중복되는 차종, 연식에 대한 정보가 있어서, 가장 최신의 정보를 이용함.
# mei = pd.merge(mei, mcr, on='MTD_ID')  # 조공서 정보와, 차종, 년식에 대한 정보를 얻기 위해서

mendi = pd.read_csv(os.path.join(data_dir, 'CRDN_EPWSMCDMENDI.csv'))
mendi = mendi.loc[mendi['LVL'] == 1.0] # LEVEL 1인 item만 가지고 일단 진행

def mendi_item_no_format_change():
    missed = 0.0
    for idx, row in mendi.iterrows():
        item_no = row['ITEM_NO']
        pattern = re.search('([A-Za-z\d]{5})[- ]?([A-Za-z\d]{5,6})', item_no)   ###
        # MENDI에서 UPGO_N 에서는 전체 10개먄, 뒤에 5개를 쓰면되고
        # MENDI에서 UPGO_N 에서는 전체 12개면, 뒤에 6개와 맨뒤에 하나 E가 붙는데, 이 E를 빼면 UPG 매칭 댐
        if pattern:
            item_no = pattern.group(1) + '-' + pattern.group(2)
            mendi.loc[idx, 'ITEM_NO'] = item_no
        else:
            mendi.drop(idx, inplace=True) ### 안맞음녀 제외해서 뺀다.
            missed += 1.0
        if idx % 10000 == 0:
            print('missed percentage of item_no of mendi', missed / len(mendi))

    mendi.to_csv(os.path.join(data_dir, 'mendi_format_changed.csv'), index=False)


e_hw = pd.read_csv(os.path.join(data_dir, 'e_hw_updated.csv'))

e_hw.loc[e_hw['ITEM_NO'].isnull(), 'ITEM_NO'] = [list() for _ in range(sum(e_hw['ITEM_NO'].isnull()))]

e_hw['ITEM_NO'] = e_hw['ITEM_NO'].apply(lambda x: [x] if x else x)

e_hw_nm_en = e_hw.loc[e_hw['ITEM_NM_EN'].notnull(), ['ITEM_NM_EN', 'E_HW']]
e_hw_nm_ko = e_hw.loc[e_hw['ITEM_NM_KO'].notnull(), ['ITEM_NM_KO', 'E_HW']]



def get_item_no(val_1, val_2):

    pattern = re.search('([A-Za-z\d]{5})[- ]?([A-Za-z\d]{5,6})', val_1)
    pattern_desc = re.search('([A-Za-z\d]{5})[- ]?([A-Za-z\d]{5,6})', val_2)
    if pattern:
        group_2 = re.sub('[- ]', '', pattern.group(2))
        item_no = pattern.group(1) + '-' + group_2

    elif pattern_desc:
        group_2 = re.sub('[- ]', '', pattern_desc.group(2))
        item_no = pattern_desc.group(1) + '-' + group_2
    else:
        item_no = ''
    return item_no

def get_part_type(item_nm, name_type):
    if name_type == 'ITEM_NM_KO':
        result = e_hw_nm_ko.loc[e_hw_nm_ko[name_type] == item_nm, 'E_HW'].unique()
    elif name_type == 'ITEM_NM_EN':
        result = e_hw_nm_en.loc[e_hw_nm_en[name_type] == item_nm, 'E_HW'].unique()
    return result

def initial_search_part_type(mei, e_hw, mtd_list):


    unmatched_item_no_list = list()


    def add_item_no_to_e_hw(item_no, item_nm, nm_type, e_hw):
        multiple_matches = e_hw.loc[
            e_hw[nm_type] == item_nm, 'ITEM_NO'].values
        for value in multiple_matches:
            value.append(item_no)


    for mtd_id in mtd_list:
        subset = mei.loc[mei['MTD_ID'] == mtd_id]

        for idx, row in subset.iterrows():
            item_no = str(row['ITEM_NO'])
            item_desc = str(row['APLY_DESC'])
            item_nm_ko = str(row['ITEM_NM_KO'])
            item_nm_en = str(row['ITEM_NM_EN'])

            item_no = get_item_no(item_no, item_desc)

            if item_no:
                ko_match = get_part_type(item_nm_ko, 'ITEM_NM_KO')
                en_match = get_part_type(item_nm_en, 'ITEM_NM_EN')

                if len(ko_match) > 0 and len(en_match) > 0:
                    assert set(ko_match) == set(
                        ko_match), 'english type and korean type does not match'

                if len(ko_match) > 0:
                    assert len((ko_match)) == 1, 'more than one type of part type'

                    add_item_no_to_e_hw(item_no, item_nm_ko, 'ITEM_NM_KO', e_hw)

                if len(en_match) > 0:
                    assert len((en_match)) == 1, 'more than one type of part type'
                    add_item_no_to_e_hw(item_no, item_nm_ko, 'ITEM_NM_KO', e_hw)


                if len(ko_match) == 0 and len(en_match) == 0:
                    print('no match', idx, item_nm_ko, item_nm_en)
                    unmatched_item_no_list.append((mtd_id, item_no, item_nm_ko,
                                                   item_nm_en))


    return unmatched_item_no_list, e_hw

def output_data_generation(mei, unmatched_item_no_list, e_hw, default_E, mtd_list):

    def check_e_hw_multiple_match(value, e_hw):
        value = str(value)
        exact_match = e_hw['ITEM_NO'].apply(lambda item_no_list:
                                            np.array(
                                                [True if value in x else False for x in
                                                 item_no_list]).any())
        match = e_hw['ITEM_NO'].apply(lambda item_no_list:
                                      np.array(
                                          [True if value.split('-')[0] in x.split('-')[
                                              0] else False for x in item_no_list]).any())
        e_hw_values = e_hw.loc[match, 'E_HW'].unique()
        exact_e_hw_values = e_hw.loc[exact_match, 'E_HW'].unique()

        if len(exact_e_hw_values) > 1:
            print('exact multiple')
            return None
        elif len(exact_e_hw_values) == 1:
            print('matched exactly from eb_hw db')
            return exact_e_hw_values
        elif len(e_hw_values) > 1:
            print('multiple')
            return None
        elif len(e_hw_values) == 1:
            print('matched from eb_hw db')
            return e_hw_values
        else:
            assert len(exact_e_hw_values) + len(e_hw_values) == 0
            print('no match at all')
            return None

    def update_list(item_no, item_type, eitem_list, hw_list):

        if item_type == 'hw':
            hw_list.append(item_no)
        elif item_type == 'E':
            eitem_list.append(item_no)


    print('output data generation')
    output_data = pd.DataFrame(columns=['MTD_ID', 'EITEM_LIST', 'HW_LIST'])

    unmatched_item_no_list = list()


    for mtd_id in mtd_list:
        subset = mei.loc[mei['MTD_ID'] == mtd_id]

        eitem_list = list()
        hw_list = list()

        for idx, row in subset.iterrows():
            item_no = str(row['ITEM_NO'])
            item_desc = str(row['APLY_DESC'])
            item_nm_ko = str(row['ITEM_NM_KO'])
            item_nm_en = str(row['ITEM_NM_EN'])

            item_no = get_item_no(item_no, item_desc)

            if item_no:
                ko_match = get_part_type(item_nm_ko, 'ITEM_NM_KO')
                en_match = get_part_type(item_nm_en, 'ITEM_NM_EN')

                if len(ko_match) > 0 and len(en_match) > 0:
                    assert set(ko_match) == set(
                        ko_match), 'english type and korean type does not match'

                if len(ko_match) > 0:
                    assert len(set(ko_match)) == 1, 'more than one type of part type'
                    update_list(item_no, ko_match, eitem_list, hw_list)

                elif len(en_match) > 0:
                    assert len(set(en_match)) == 1, 'more than one type of part type'
                    update_list(item_no, en_match, eitem_list, hw_list)

                else:
                    search_result = check_e_hw_multiple_match(item_no, e_hw)

                    if search_result:
                        update_list(item_no, search_result, eitem_list, hw_list)
                    elif default_E:
                        update_list(item_no, 'E', eitem_list, hw_list)

                    # else:
                    #     print('failed to found match from e_hw database from item_no rule')
                    #     unmatched_item_no_list.append((mtd_id, item_no, item_nm_ko,
                    #                                    item_nm_en))

            else:
                print('no item_no')

        eitem_list = list(pd.Series(eitem_list).unique())
        hw_list = list(pd.Series(hw_list).unique())
        insert_row = pd.Series([mtd_id, eitem_list, hw_list],
                        index=['MTD_ID', 'EITEM_LIST', 'HW_LIST'])
        # meta_info = subset.iloc[0][['CAR', 'CAR_YEAR', 'UPG_NO']]
        # insert_row = insert_row.append(meta_info)
        output_data = output_data.append(insert_row, ignore_index=True)

    # output_data.drop_duplicates
    # print('unmatched_item_no_list output as excel file')
    # pd.DataFrame(unmatched_item_no_list).to_csv('unmatched_item_no.xlsx', index=False)

    print('output data split')
    return output_data

def split_output_data(output_data):
    output_split_by_eitem = pd.DataFrame(columns=output_data.columns)

    cols_except_mtd_id = list(output_data.columns)
    cols_except_mtd_id.remove('MTD_ID')
    for i, idx in enumerate(output_data.index):
        if i % 1000 == 0:
            print(i / len(output_data))

        row = output_data.loc[idx]
        mtd_id = row['MTD_ID']
        eitem_list = row['EITEM_LIST']
        # hw_list = row['HW_LIST']
        if len(eitem_list) > 1:
            for e_item_idx, eitem in enumerate(eitem_list):
                new_mtd_id = pd.Series(str(int(mtd_id)) + '_' + str(e_item_idx), index=['MTD_ID'])
                new_row = row[cols_except_mtd_id]
                new_row['EITEM_LIST'] = eitem
                output_split_by_eitem.loc[len(output_split_by_eitem)] = new_row.append(
                    new_mtd_id)
        else:
            try:
                row['EITEM_LIST'] = eitem_list.pop()
            except:
                row['EITEM_LIST'] = list()
            output_split_by_eitem.loc[len(output_split_by_eitem)] = row

    # print('output_data saved as output_data.csv')
    # output_split_by_eitem.to_csv('output_data_split.csv', index=False)
    output_split_by_eitem.rename(columns={'EITEM_LIST' : 'E_ITEM'}, inplace=True)
    return output_split_by_eitem

def get_upg_no_by_car_year(remove_trailing_U):
    if remove_trailing_U:
        print('trailing U remove')
    car_year = mupg[['PROJ_CCAR', 'PROJ_CYY']].drop_duplicates()
    upg_no_by_car = dict()
    for idx, (car, year) in car_year.iterrows():
        if remove_trailing_U:
            if car.endswith('U') and len(car) > 2:
                continue
        # car = re.sub('\s', '', car)
        upg_no_by_car[(car, year)] = list()
        car_cond = mupg['PROJ_CCAR'] == car
        year_cond = mupg['PROJ_CYY'] == year
        upg_no_data = mupg.loc[car_cond & year_cond, 'UPGO_N']
        if sum(upg_no_data.duplicated()) != 0:
            print('error')
        for idx, upg_no in enumerate(upg_no_data):
            upg_no = re.sub('\s', '', upg_no)
            match = re.search('([A-Za-z\d]{2})UPG([A-Za-z\d]{5,6})', upg_no)
            assert match, 'mupg upgo_n not matching pattern, {}'.format(upg_no)
            assert match.group(1) == car, '{}_{}_{}'.format(match.group(1), car, upg_no)
            assert not  match.group(2) in upg_no_by_car[(car, year)], \
                '{}_{}'.format(upg_no, match.group(2))
            upg_no_by_car[(car, year)].append(match.group(2))

    return upg_no_by_car

def get_mtd_by_car_year(car, year, upg_no_by_car, output_data):

    upg_no_list = upg_no_by_car[(car, year)]
    # print(upg_no_list)
    cond = list()
    print(car, year, 'start')
    for upg_no in upg_no_list:
        bool_idx = np.where(output_data['UPG_NO'].apply(lambda x: x == upg_no[:6]))[0]
        assert pd.Series([i not in cond for i in  bool_idx]).all()
        cond.extend(bool_idx)

    subset = output_data.loc[cond]
    subset.to_pickle('output_combined_{}_{}.df'.format(car, year))
    output_data_split = split_output_data(subset)
    output_data_split['CAR'] = car
    output_data_split['CAR_YEAR'] = year
    output_data_split.to_pickle('output_{}_{}.df'.format(car, year))
    print(car, year, 'done')
    return output_data_split

def get_output_data():
    unique_mtd = mei['MTD_ID'].unique()
    mtd_list_split = np.split(unique_mtd,
                              np.arange(0, len(unique_mtd), len(unique_mtd) // 12)[1:])

    if not os.path.exists(os.path.join(data_dir, 'e_hw_new.df')):
        print('e_hw update')
        pool = mp.Pool(processes=12)
        results = [pool.apply_async(initial_search_part_type, args=(mei, e_hw, mtd_list))
                   for mtd_list in
                   mtd_list_split]
        result_get = [p.get() for p in results]


        unmatched_item_no_list = list()
        e_hw_tmp_list = list()
        for val in result_get:
            unmatched_tmp, e_hw_tmp = val
            unmatched_item_no_list.extend(unmatched_tmp)
            e_hw_tmp_list.append(e_hw_tmp)

        for e_hw_tmp in e_hw_tmp_list:
            for idx, row in e_hw_tmp.iterrows():
                e_hw.loc[idx, 'ITEM_NO'].extend(row['ITEM_NO'])
                e_hw.loc[idx, 'ITEM_NO'] = list(set(e_hw.loc[idx, 'ITEM_NO']))

        e_hw.to_pickle(os.path.join(data_dir, 'e_hw_new.df'))
    else:
        print('fetching e_hw')
        e_hw = pd.read_pickle(os.path.join(data_dir, 'e_hw_new.df'))


    # df = pd.DataFrame(unmatched_item_no_list, columns=['MTD_ID', 'ITEM_NO',
    #                                                    'ITEM_NM_KO', 'ITEM_NM_EN'])
    # df.drop_duplicates(inplace=True)
    # df.to_csv('unmatched_item_no.csv', index=False)

    unmatched_item_no_list = list() ## TODO

    pool = mp.Pool(processes=12)
    results = [pool.apply_async(output_data_generation,
                                args=(mei, unmatched_item_no_list, e_hw, True, mtd_list))
               for mtd_list in
               mtd_list_split]

    output_data = pd.concat([p.get() for p in results])

    output_data.to_pickle('duplicated_output.df')

    output_data = pd.merge(output_data,
                   mtd[['MTD_ID', 'UPG_NO']], on='MTD_ID')

    # output_data.to_csv('output_data.csv', index=False)
    return output_data

def make_output_by_car_year():


    if os.path.exists('output_data.df'):
        output_data = pd.read_pickle('output_data.df')
    else:
        output_data = get_output_data()
        output_data.to_pickle('output_data.df')

    upg_no_by_car = get_upg_no_by_car_year(True)

    # get_mtd_by_car_year('UG', 15, upg_no_by_car, output_data)

    pool = mp.Pool(processes=12)
    results = [pool.apply_async(get_mtd_by_car_year, args=(car, year, upg_no_by_car,
                                                           output_data))
               for car, year in
               upg_no_by_car.keys()]

    output_data = pd.concat([p.get() for p in results])
    output_data.to_pickle('output_data_by_car_year.df')


def split_input_data_eitem_hw():
    if os.path.exists(os.path.join(data_dir, 'mendi_format_changed.csv')):
        print('fetching mendi_format_changed.csv')
        mendi = pd.read_csv(os.path.join(data_dir, 'mendi_format_changed.csv'))
    else:
        print('making mendi_format_changed.csv')
        mendi = mendi_item_no_format_change()
    upg_by_car_year = get_upg_no_by_car_year(True)
    pool = mp.Pool(processes=12)
    results = [pool.apply_async(get_input_data, args=(mendi, car, year, upg_list,))
               for (car, year), upg_list in upg_by_car_year.items()]

    # a = get_input_data('UG', 15, upg_by_car_year[('UG', 15)])
    get_results = [p.get() for p in results]
    # output_data = pd.concat(get_results)


def get_input_data(mendi, car, year, upg_list):
    data = pd.DataFrame(columns=['UPG_NO', 'E_LIST', 'HW_LIST'])
    for i, upg in enumerate(upg_list):
        # if i % 100 == 0:
        print(car, year, i / len(upg_list))
        subset = mendi.loc[mendi['UPG'] == upg]

        if  len(subset) == 0:
            print('upg in mupgo but not in mendi')
            continue

        item_no_list = list(subset['ITEM_NO'])
        assert len(item_no_list) > 0, upg
        eitem_list = list()
        hw_list = list()

        for item_no in item_no_list:
            cond = e_hw.loc[e_hw['ITEM_NO'].apply(lambda x: item_no in x),
                              'E_HW'].unique()
            if len(cond) == 1:
                cond = cond[0]
                if cond == 'hw':
                    hw_list.append(item_no)
                elif cond == 'e':
                    eitem_list.append(item_no)
            elif len(cond) > 1:
                assert len(cond) < 2
                continue
            elif len(cond) == 0:
                eitem_list.append(item_no)
        data = data.append(
            pd.Series([upg, eitem_list, hw_list], index=['UPG_NO', 'E_LIST',
                                                         'HW_LIST']),
            ignore_index=True)
    data['CAR'] = car
    data['CAR_YEAR'] = year
    data.to_pickle('input_{}_{}.df'.format(car, year))
    return data

if __name__ == '__main__':
    # make_output_by_car_year()
    split_input_data_eitem_hw()
