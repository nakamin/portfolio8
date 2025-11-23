import jpholiday

def make_date(df):
    """
    - timestamp列からmonth, hour,is_holiday列を追加する
    """

    df["month"] = df["timestamp"].dt.month
    df["hour"] = df["timestamp"].dt.hour
    
    df["is_holiday"] = df["timestamp"].dt.date.apply(jpholiday.is_holiday)
    
    def is_year_end(date):
        return (date.month == 12 and date.day >= 29) or (date.month == 1 and date.day <= 3)

    df["is_year_end"] = df["timestamp"].dt.date.apply(is_year_end)
    
    def is_obon(date):
        return date.month == 8 and date.day in [13, 14, 15]

    df["is_obon"] = df["timestamp"].dt.date.apply(is_obon)
    df["is_holiday"] = df["is_holiday"] | df["is_year_end"] | df["is_obon"]
    df["is_holiday"] = df["is_holiday"].astype(int)
    
    df.drop(columns=["is_year_end", "is_obon"], inplace=True)
    
    return df

def make_daypart(hour: int) -> str:
    """時間を6つに分ける"""
    if 5 <= hour <= 7:  return "dawn"
    if 8 <= hour <= 11: return "morning"
    if 12 <= hour <= 13:return "noon"
    if 14 <= hour <= 16:return "afternoon"
    if 17 <= hour <= 19:return "evening"
    return "night"

def make_season(month: int) -> str:
    """月を季節ごとに4つに分ける"""
    if month in (3,4,5):   return "spring"
    if month in (6,7,8):   return "summer"
    if month in (9,10,11): return "autumn"
    return "winter"