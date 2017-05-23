import os
import re

import numpy as np
import pandas as pd

data_dir = '/home/jinwon/PycharmProjects/hyundai/data'

mei = pd.read_csv(os.path.join(data_dir, 'CRDN_EPWSMCDMCMEI.csv'), encoding='cp949')
mtd = pd.read_csv(os.path.join(data_dir, 'CRDN_EPWSMCDMCMTD.csv'), encoding='cp949')

e_hw = pd.read_csv(os.path.join(data_dir, 'e_hw.csv'))

e_hw['ITEM_NO'] = None
e_hw['ITEM_NO'] = e_hw['ITEM_NO'].apply(lambda x: [])

e_hw_nm_en = e_hw.loc[e_hw['ITEM_NM_EN'].notnull(), ['ITEM_NM_EN', 'E_HW']]
e_hw_nm_ko = e_hw.loc[e_hw['ITEM_NM_KO'].notnull(), ['ITEM_NM_KO', 'E_HW']]


def get_item_no(val_1, val_2):
    pattern = re.search('([A-z\d]{5})[- ]*([A-z\d]{5,})', val_1)
    pattern_desc = re.search('([A-z\d]{5})[- ]*([A-z\d]{5,})', val_2)
    if pattern:
        item_no = pattern.group(1) + '-' + pattern.group(2)
    elif pattern_desc:
        item_no = pattern_desc.group(1) + '-' + pattern_desc.group(2)
    else:
        item_no = ''
    return item_no


def get_part_type(item_nm, name_type):
    if name_type == 'ITEM_NM_KO':
        result = e_hw_nm_ko.loc[e_hw_nm_ko[name_type] == item_nm, 'E_HW']
    elif name_type == 'ITEM_NM_EN':
        result = e_hw_nm_en.loc[e_hw_nm_en[name_type] == item_nm, 'E_HW']
    return result, result.unique()


def update_list(item_no, item_type, eitem_list, hw_list, hw_unique_list):

    item_type = item_type.unique().values

    if item_type == 'hw':
        hw_list.append(item_no)
    elif item_type == 'hw_unique':
        hw_unique_list.append(item_no)
    elif item_type == 'E':
        eitem_list.append(item_no)


def add_item_no_to_e_hw(item_no, item_nm, nm_type, e_hw=e_hw):
    multiple_matches = e_hw.loc[
        e_hw[nm_type] == item_nm, 'ITEM_NO'].values
    for value in multiple_matches:
        value.append(item_no)


def initial_search_part_type(e_hw):
    # output_data = pd.DataFrame(columns=['MTD_ID', 'EITEM_LIST', 'HW_LIST', 'HW_UNIQUE_LIST'])
    unique_mtd = mei['MTD_ID'].unique()
    unmatched_item_no_list = list()


    for mtd_id in unique_mtd:
        subset = mei.loc[mei['MTD_ID'] == mtd_id]
        #     item_no_list = subset['ITEM_NO']
        #     item_desc_list = subset['APLY_DESC']
        #     item_nm_en_list= subset['ITEM_NM_EN']
        #     item_nm_ko_list = subset['ITEM_NM_KO']

        eitem_list = list()
        hw_list = list()
        hw_unique_list = list()


        for idx, row in subset.iterrows():
            item_no = str(row['ITEM_NO'])
            item_desc = str(row['APLY_DESC'])
            item_nm_ko = str(row['ITEM_NM_KO'])
            item_nm_en = str(row['ITEM_NM_EN'])

            item_no = get_item_no(item_no, item_desc)

            if item_no:
                ko_match = get_part_type(item_nm_ko, 'ITEM_NM_KO')
                en_match = get_part_type(item_nm_en, 'ITEM_NM_EN')

                if len(ko_match) > 1:
                    assert len(set(ko_match)) == 1, 'more than one type of part type'
                    print('over 1 match, ko', item_nm_ko, item_nm_en)
                    add_item_no_to_e_hw(item_no, item_nm_ko, 'ITEM_NM_KO')

                if len(en_match) > 1:
                    assert len(set(en_match)) == 1, 'more than one type of part type'
                    add_item_no_to_e_hw(item_no, item_nm_ko, 'ITEM_NM_KO')
                    print('over 1 match, en', item_nm_ko, item_nm_en)

                    if len(ko_match) > 1:
                        assert set(ko_match) == set(ko_match), 'english type and korean type does not match'

                if len(ko_match) == 0:
                    if len(en_match) == 0:
                        print('no match', idx, item_nm_ko, item_nm_en)
                        unmatched_item_no_list.append((mtd_id, item_no, item_nm_ko,
                                                       item_nm_en))

                    elif len(en_match) == 1:
                        add_item_no_to_e_hw(item_no, item_nm_en, 'ITEM_NM_EN')
                        # update_list(item_no, en_match, eitem_list, hw_list, hw_unique_list)

                elif len(ko_match) == 1:
                    if len(en_match) == 0:
                        add_item_no_to_e_hw(item_no, item_nm_ko, 'ITEM_NM_KO')
                        # update_list(item_no, ko_match, eitem_list, hw_list, hw_unique_list)

                    elif len(en_match) == 1:
                        if bool((ko_match == en_match).all()):
                            add_item_no_to_e_hw(item_no, item_nm_ko, 'ITEM_NM_KO') # 한글 영어 둘중 아무거나 써도 됨
                            # update_list(item_no, en_match, eitem_list, hw_list, hw_unique_list)
                        else:
                            print('disagreed', item_nm_ko, item_nm_en)
            else:
                print('no item_no')
        # row = pd.Series([mtd_id, eitem_list, hw_list, hw_unique_list],
        #                 index=['MTD_ID', 'EITEM_LIST', 'HW_LIST', 'HW_UNIQUE_LIST'])
        # output_data = output_data.append(row, ignore_index=True)
    return unmatched_item_no_list


def check_e_hw_multiple_match(value):
    value = str(value)
    exact_match = match = e_hw['ITEM_NO'].apply(lambda item_no_list:
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
        return exact_e_hw_values
    elif len(e_hw_values) > 1:
        print('multiple')
        return None
    elif len(e_hw_values) == 1:
        return e_hw_values
    else:
        assert len(exact_e_hw_values) + len(e_hw_values) == 0
        print('no match at all')
        return None


def output_data_generation(unmatched_item_no_list):
    output_data = pd.DataFrame(columns=['MTD_ID', 'EITEM_LIST', 'HW_LIST', 'HW_UNIQUE_LIST'])
    unique_mtd = mei['MTD_ID'].unique()
    unmatched_item_no_list = list()

    for mtd_id in unique_mtd:
        subset = mei.loc[mei['MTD_ID'] == mtd_id]

        eitem_list = list()
        hw_list = list()
        hw_unique_list = list()

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
                    update_list(item_no, ko_match, eitem_list, hw_list, hw_unique_list)

                elif len(en_match) > 0:
                    assert len(set(en_match)) == 1, 'more than one type of part type'
                    update_list(item_no, en_match, eitem_list, hw_list, hw_unique_list)

                else:
                    search_result = check_e_hw_multiple_match(item_no)

                    if search_result:
                        update_list(item_no, search_result, eitem_list, hw_list, hw_unique_list)
                    else:
                        print('failed to found match from e_hw database from item_no rule')
                        unmatched_item_no_list.append((mtd_id, item_no, item_nm_ko,
                                                       item_nm_en))

            else:
                print('no item_no')
        insert_row = pd.Series([mtd_id, eitem_list, hw_list, hw_unique_list],
                        index=['MTD_ID', 'EITEM_LIST', 'HW_LIST', 'HW_UNIQUE_LIST'])
        output_data = output_data.append(insert_row, ignore_index=True)
    return output_data, unmatched_item_no_list


unmatched_item_no_list = initial_search_part_type(e_hw)

output_data, unmatched_item_no_list = output_data_generation(unmatched_item_no_list)

print(output_data)

print(unmatched_item_no_list)