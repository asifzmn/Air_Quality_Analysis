import sys; sys.path.insert(0,".")
import pandas as pd
from data_preparation.additional_data_preparation import get_mobility_data
from data_preparation.spatio_temporal_filtering import read_bd_data
from exploration import yearly_seasonal_decomposition_bd
from data_preparation.handling_missing_entries import FillMissingFromCombined
cs=read_bd_data()[5]
ts=cs.apply(FillMissingFromCombined).iloc[:,0]["2017-03-01":"2022-02-28"].resample("D").mean()
ts=yearly_seasonal_decomposition_bd(ts); ts=(ts-ts.min()).iloc[:,0]
mob=get_mobility_data()
low=ts.index.isin(mob.index[(mob.workplaces<0)&(mob.transit_stations<0)])
inmob=ts.index>=mob.index.min()
for name,other in [("other days, mobility period (Feb 2020-Feb 2022)",(~low)&inmob),("all other days 2017-2022",~low),("pre-COVID days (before Mar 2020)",ts.index<"2020-03-01")]:
    a,b=ts[low].median(),ts[other].median()
    print(f"{name}: low={a:.2f} other={b:.2f} diff={b-a:.2f} | means diff={ts[other].mean()-ts[low].mean():.2f}")
ts.to_frame("pm").assign(low=low).to_csv("desea_series.csv")
