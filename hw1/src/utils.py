import pandas as pd


class DataProcessor:

    @staticmethod
    def read_data(filename):
        df = pd.read_csv(filename,
                         delim_whitespace=True,
                         header=None)
        sentence_ids = \
            list(map(lambda x: '_'.join(x.split('_')[:-1]), df[0]))
        frame_ids = \
            list(map(lambda x: int(x.split('_')[-1]), df[0]))
        del df[0]

        df['sid'] = sentence_ids
        df['fid'] = frame_ids
        df = df.pivot('sid', 'fid')

        # swap and sort so the first level is fid and
        # the second is 69 features
        df.columns = df.columns.swaplevel(0, 1)
        df.sort_index(axis=1, level=0, inplace=True)

        # sort by sentence id
        df.sort_index(axis=0, inplace=True)

        return df.as_matrix().reshape(df.shape[0],
                                      df.columns[-1][0],
                                      df.columns[-1][1])

    @staticmethod
    def read_label(filename):
        df = pd.read_csv(filename,
                         header=None)
        sentence_ids = \
            list(map(lambda x: '_'.join(x.split('_')[:-1]), df[0]))
        frame_ids = \
            list(map(lambda x: int(x.split('_')[-1]), df[0]))
        del df[0]

        df['sid'] = sentence_ids
        df['fid'] = frame_ids
        df = df.pivot('sid', 'fid')

        # sort by sentence id
        df.sort_index(axis=0, inplace=True)

        return df.as_matrix()
