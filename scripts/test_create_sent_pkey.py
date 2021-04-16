import pandas as pd


df = pd.DataFrame({})

df['A'] = ['a', 'a', 'b', 'c', 'c', 'd', 'e', 'e', 'e', 'e']


def create_pkey(df, colname):
    """
    Function to create sentence primary key, which is a function of the accession number
    and count of sentence.
    """
    # Create Primary Key Object.  Initialize = 1
    pkey = []
    count = 1
    # Iterate Column values as pair
    for val1, val2 in zip(df[colname].values.tolist(),
                          df[colname].values.tolist()[1:]):
        # If Value 2 == Value 1, Increase Count and append to pkey list.
        if val2 == val1:
            count += 1
            pkey.append(count)
        # When Value 2 != Value 1 then reset count & append.
        else:
            count = 1
            pkey.append(count)
    # Add Primary Key To DataFrame
    df['sent_pkey'] = pkey
    # Return df
    return df

