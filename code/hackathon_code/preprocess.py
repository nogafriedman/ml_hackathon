import pandas as pd

BOOLEAN_COLUMNS = ['guest_is_not_the_customer',
                   'guest_is_not_the_customer',
                   'is_user_logged_in',
                   'is_first_booking']

DATE_COLUMNS = ['checkin_date', 'checkout_date']
DATETIME_COLUMNS = ['booking_datetime', 'hotel_live_date']

# 4 empty,
CATEGORY_COLUMNS = ['hotel_country_code', 'accommadation_type_name', 'customer_nationality',
                    'guest_nationality_country_name', 'origin_country_code', 'language', 'original_payment_method',
                    'original_payment_type', 'original_payment_currency', 'hotel_area_code',
                    'hotel_brand_code', 'hotel_chain_code', 'hotel_city_code', 'request_nonesmoke',
                    'request_latecheckin', 'request_highfloor', 'request_largebed', 'request_twinbeds',
                    'request_airport', 'request_earlycheckin']

# TBD what to do with it...
CATEGORY_COLUMNS_WITH_TOO_MANY_UNIQUE_VALUES = ['hotel_id', 'hotel_area_code',
                                                'hotel_city_code']
DROP_BECAUSE_MANY_UNFILLED_ROWS = ['hotel_brand_code', 'hotel_chain_code']
IRRELEVANT_COLUMNS = ['h_booking_id', 'h_customer_id']

DROPPED_COLUMNS = CATEGORY_COLUMNS_WITH_TOO_MANY_UNIQUE_VALUES + DROP_BECAUSE_MANY_UNFILLED_ROWS

CATEGORY_COLUMNS_TO_EXPAND = [col for col in CATEGORY_COLUMNS if
                              col not in DROPPED_COLUMNS]


# func that returns a dummified DataFrame of significant dummies in a given column
def limit_categories(df, column_names):
    threshold = 0.1

    final_df = df.copy()

    final_df[column_names] = final_df[column_names].astype('str')

    for column in column_names:
        # removes the bind
        column_vals = final_df[column]

        # what is the ratio of a dummy in whole column
        count = pd.value_counts(column_vals) / len(column_vals)

        # cond whether the ratios is higher than the threshold
        mask = column_vals.isin(count[count > threshold].index)

        # replace the ones which ratio is lower than the threshold by a special name
        column_vals[~mask] = "OTHERS"

        final_df[column] = column_vals

    return final_df


def categorize_columns(df, column_names):
    final_df = df.copy()
    final_df[column_names] = final_df[column_names].astype('str')

    for column_name in column_names:
        # convert to category using get_dummies
        final_df = pd.get_dummies(final_df, columns=[column_name], prefix=f'cat_{column_name}', prefix_sep='_')

    # final_df = final_df.drop(column_names, axis=1)
    return final_df


def parse_cancellation_policy(df):
    final_df = df.copy()

    final_df['total_nights'] = (final_df['checkout_date'] - final_df['checkin_date']).dt.days

    policies = final_df['cancellation_policy_code'].str.split('_', expand=True)

    for policy in policies.columns:
        final_df[f'policy{policy}_days_before_checkin'] = policies[policy].str.extract(r'(\d+)(?=D)').astype(
            float)
        # if no days count, it's 0:
        final_df[f'policy{policy}_days_before_checkin'] = final_df[f'policy{policy}_days_before_checkin'].fillna(0)

        final_df[f'policy{policy}_penalty_percents'] = policies[policy].str.extract(
            r'(\d+)(?=P)').astype(float) / 100
        final_df[f'policy{policy}_penalty_nights'] = policies[policy].str.extract(r'(\d+)(?=N)').astype(
            float)

        # now calculate percents to be as nights:
        final_df[f'policy{policy}_penalty_nights'] = final_df[f'policy{policy}_penalty_nights'].fillna(
            final_df[f'policy{policy}_penalty_percents'] * final_df[
                'total_nights'])

        # finaly if no policy, get the previous one:
        if policy > 0:
            final_df[f'policy{policy}_penalty_nights'] = final_df[f'policy{policy}_penalty_nights'].fillna(
                final_df[f'policy{policy - 1}_penalty_nights'])

        final_df = final_df.drop([f'policy{policy}_penalty_percents'], axis=1)

    final_df = final_df.drop(['cancellation_policy_code'], axis=1)

    return final_df


def replace_empty_with_OTHERS(df, column_names):
    final_df = df.copy()
    for column_name in column_names:
        # Replace the empty values with UNKNOWN
        final_df[column_name] = final_df[column_name].fillna("OTHERS")

    return final_df


def split_date_to_days_since_2000_and_minutes_since_start_of_day(df, column_names):
    final_df = df.copy()

    for column_name in column_names:
        final_df[column_name] = pd.to_datetime(final_df[column_name])
        # Convert the column to  number of days since beginning of the year
        final_df[f"{column_name}_days_since_2000"] = (final_df[column_name] - pd.to_datetime("2000-01-01")).dt.days

        # Convert the column to minutes since start of day
        final_df[f"{column_name}_minutes_since_start_of_day"] = final_df[column_name].dt.hour * 60 + final_df[
            column_name].dt.minute

    return final_df


def split_date_to_days_since_2000(df, column_names):
    final_df = df.copy()
    for column_name in column_names:
        final_df[column_name] = pd.to_datetime(final_df[column_name])
        # Convert the column to  number of days since beginning of the year
        final_df[f"{column_name}_days_since_2000"] = (final_df[column_name] - pd.to_datetime("2000-01-01")).dt.days

    return final_df


def convert_cancellation_date_to_did_cancel(df):
    final_df = df.copy()

    # Convert the column to 0 and 1
    final_df["did_cancel"] = final_df["cancellation_datetime"].notnull().astype(int)
    return final_df


def fix_charge_option(df, column_name):
    final_df = df.copy()

    # Replace the values
    final_df[column_name] = final_df[column_name].replace({"Pay Now": 0, "Pay Later": 1, "Pay at Check-in": 1})

    return final_df


def convert_boolean_columns(df, column_names):
    final_df = df.copy()
    for column_name in column_names:
        # Convert the column to 0 and 1
        final_df[column_name] = final_df[column_name].astype(int)

    return final_df


def convert_date_column_to_days_and_time_of_day(df, column_name):
    final_df = df.copy()

    # Convert the column to  number of days since beginning of the year
    final_df[column_name] = pd.to_datetime(final_df[column_name])


NUMERICAL_COLUMNS = ['no_of_adults',
                     'no_of_children',
                     'no_of_extra_bed',
                     'no_of_room']


def get_preprocessed_data(df, task=1, is_test=False):
    if task == 1 and not is_test:
        df = df.pipe(convert_cancellation_date_to_did_cancel)
    if task == 2 and not is_test:
        df = df.pipe(split_date_to_days_since_2000_and_minutes_since_start_of_day, ['cancellation_datetime'])

    df = (df
          # Handle boolean columns
          .pipe(fix_charge_option, "charge_option")
          .pipe(convert_boolean_columns, BOOLEAN_COLUMNS)
          )

    df = df.drop(IRRELEVANT_COLUMNS, axis=1)

    # Show boolean columns for inspection...
    # for col in BOOLEAN_COLUMNS:
    #     print(col)
    #     print(df[col].value_counts())

    # Handle date columns
    df = (df
          .pipe(split_date_to_days_since_2000_and_minutes_since_start_of_day, DATETIME_COLUMNS)
          .pipe(split_date_to_days_since_2000, DATE_COLUMNS)
          )

    df = df.drop(CATEGORY_COLUMNS_WITH_TOO_MANY_UNIQUE_VALUES + DROP_BECAUSE_MANY_UNFILLED_ROWS, axis=1)

    # Handle categorical columns
    df = (df
          .pipe(replace_empty_with_OTHERS, CATEGORY_COLUMNS_TO_EXPAND)
          .pipe(limit_categories, CATEGORY_COLUMNS_TO_EXPAND)
          .pipe(categorize_columns, CATEGORY_COLUMNS_TO_EXPAND)
          )

    df = df.pipe(parse_cancellation_policy)

    df = df.drop(DATE_COLUMNS + DATETIME_COLUMNS, axis=1)
    df = df.drop(['policy2_days_before_checkin'], axis=1)
    if not is_test:
        df = df.drop(['cancellation_datetime'], axis=1)

    df = df.fillna(df.mean())
    return df
