import streamlit as st
import pandas as pd
import matplotlib as plt
import numpy as np
import tensorflow as tf
import lxml
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

logo_let = Image.open('data//C-La.png')
logo_ic = Image.open('data//C-L5.png')
st.set_page_config(page_title="ClutchWare-Match",page_icon=logo_ic)


Select_H=["Atlanta Hawks","Brooklyn Nets","Boston Celtics","Charlotte Hornets","Chicago Bulls","Cleveland Cavaliers","Dallas Mavericks","Denver Nuggets",
        "Detroit Pistons","Golden State Warriors","Houston Rockets","Indiana Pacers","Los Angeles Clippers","Los Angeles Lakers","Memphis Grizzlies","Miami Heat",
        "Milwaukee Bucks","Minnesota Timberwolves","New Orleans Pelicans","New York Knicks","Oklahoma City Thunder","Orlando Magic","Philadelphia 76ers","Phoenix Suns",
        "Portland Trail Blazers","Sacramento Kings","San Antonio Spurs","Toronto Raptors","Utah Jazz","Washington Wizards"]


Select_A=["Atlanta Hawks","Brooklyn Nets","Boston Celtics","Charlotte Hornets","Chicago Bulls","Cleveland Cavaliers","Dallas Mavericks","Denver Nuggets",
        "Detroit Pistons","Golden State Warriors","Houston Rockets","Indiana Pacers","Los Angeles Clippers","Los Angeles Lakers","Memphis Grizzlies","Miami Heat",
        "Milwaukee Bucks","Minnesota Timberwolves","New Orleans Pelicans","New York Knicks","Oklahoma City Thunder","Orlando Magic","Philadelphia 76ers","Phoenix Suns",
        "Portland Trail Blazers","Sacramento Kings","San Antonio Spurs","Toronto Raptors","Utah Jazz","Washington Wizards"]

Last_G = 8


st.image(logo_let)

st.header("Match Analyzer")
st.markdown("Artificial Intelligence NBA match predictor")


tm_H=st.sidebar.multiselect("Home Team",Select_H)
tm_A=st.sidebar.multiselect("Away Team",Select_A)
Bt_ou_line=st.sidebar.text_input("Bet line Over/Under",)
b=st.sidebar.button("Analyze")

def pick_tm(tm_sl):


  if tm_sl == 'Atlanta Hawks':
        tm_f = 'ATL'

  elif tm_sl == "Brooklyn Nets":
        tm_f = 'BRK'

  elif tm_sl == "Boston Celtics":
        tm_f = 'BOS'

  elif tm_sl == "Charlotte Hornets":
        tm_f = 'CHO'

  elif tm_sl == "Chicago Bulls":
        tm_f = 'CHI'

  elif tm_sl == "Cleveland Cavaliers":
        tm_f = 'CLE'

  elif tm_sl == "Dallas Mavericks":
        tm_f = 'DAL'

  elif tm_sl == "Denver Nuggets":
        tm_f = 'DEN'

  elif tm_sl == "Detroit Pistons":
        tm_f = 'DET'

  elif tm_sl == "Golden State Warriors":
        tm_f = 'GSW'

  elif tm_sl == "Houston Rockets":
        tm_f = 'HOU'

  elif tm_sl == "Indiana Pacers":
        tm_f = 'IND'

  elif tm_sl == "Los Angeles Clippers":
        tm_f = 'LAC'

  elif tm_sl == "Los Angeles Lakers":
        tm_f = 'LAL'

  elif tm_sl == "Memphis Grizzlies":
        tm_f = 'MEM'

  elif tm_sl == "Miami Heat":
        tm_f = 'MIA'

  elif tm_sl == "Milwaukee Bucks":
        tm_f = 'MIL'

  elif tm_sl == "Minnesota Timberwolves":
        tm_f = 'MIN'

  elif tm_sl == "New Orleans Pelicans":
        tm_f = 'NOP'

  elif tm_sl == "New York Knicks":
        tm_f = 'NYK'

  elif tm_sl == "Oklahoma City Thunder":
        tm_f = 'OKC'

  elif tm_sl == "Orlando Magic":
        tm_f = 'ORL'

  elif tm_sl == "Philadelphia 76ers":
        tm_f = 'PHI'

  elif tm_sl == "Phoenix Suns":
        tm_f = 'PHO'

  elif tm_sl == "Portland Trail Blazers":
        tm_f = 'POR'

  elif tm_sl == "Sacramento Kings":
        tm_f = 'SAC'

  elif tm_sl == "San Antonio Spurs":
        tm_f = 'SAS'

  elif tm_sl == "Toronto Raptors":
        tm_f = 'TOR'

  elif tm_sl == "Utah Jazz":
        tm_f = 'UTA'

  elif tm_sl == "Washington Wizards":
        tm_f = 'WAS'

  return tm_f

def tmnm(nm1):

  if nm1 == 'Atlanta Hawks':
        nm2 = 'Atlanta'

  elif nm1 == "Brooklyn Nets":
        nm2 = 'Brooklyn'

  elif nm1 == "Boston Celtics":
        nm2 = 'Boston'

  elif nm1 == "Charlotte Hornets":
        nm2 = 'Charlotte'

  elif nm1 == "Chicago Bulls":
        nm2 = 'Chicago'

  elif nm1 == "Cleveland Cavaliers":
        nm2 = 'Cleveland'

  elif nm1 == "Dallas Mavericks":
        nm2 = 'Dallas'

  elif nm1 == "Denver Nuggets":
        nm2 = 'Denver'

  elif nm1 == "Detroit Pistons":
        nm2 = 'Detroit'

  elif nm1 == "Golden State Warriors":
        nm2 = 'Golden State'

  elif nm1 == "Houston Rockets":
        nm2 = 'Houston'

  elif nm1 == "Indiana Pacers":
        nm2 = 'Indiana'

  elif nm1 == "Los Angeles Clippers":
        nm2 = 'LA Clippers'

  elif nm1 == "Los Angeles Lakers":
        nm2 = 'LA Lakers'

  elif nm1 == "Memphis Grizzlies":
        nm2 = 'Memphis'

  elif nm1 == "Miami Heat":
        nm2 = 'Miami'

  elif nm1 == "Milwaukee Bucks":
        nm2 = 'Milwaukee'

  elif nm1 == "Minnesota Timberwolves":
        nm2 = 'Minnesota'

  elif nm1 == "New Orleans Pelicans":
        nm2 = 'New Orleans'

  elif nm1 == "New York Knicks":
        nm2 = 'New York'

  elif nm1 == "Oklahoma City Thunder":
        nm2 = 'Okla City'

  elif nm1 == "Orlando Magic":
        nm2 = 'Orlando'

  elif nm1 == "Philadelphia 76ers":
        nm2 = 'Philadelphia'

  elif nm1 == "Phoenix Suns":
        nm2 = 'Phoenix'

  elif nm1 == "Portland Trail Blazers":
        nm2 = 'Portland'

  elif nm1 == "Sacramento Kings":
        nm2 = 'Sacramento'

  elif nm1 == "San Antonio Spurs":
        nm2 = 'San Antonio'

  elif nm1 == "Toronto Raptors":
        nm2 = 'Toronto'

  elif nm1 == "Utah Jazz":
        nm2 = 'Utah'

  elif nm1 == "Washington Wizards":
        nm2 = 'Washington'

  return nm2


def AI(tm_pt, tm_av, opp_av):
    capa = tf.keras.layers.Dense(units=1, input_shape=[1])
    modelo = tf.keras.Sequential([capa])

    oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
    oculta2 = tf.keras.layers.Dense(units=3)
    salida = tf.keras.layers.Dense(units=1)
    modelo = tf.keras.Sequential([oculta1, oculta2, salida])
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(0.01),
        loss='mean_squared_error'
    )

    print("Comenzando entrenamiento...")
    historial = modelo.fit(tm_av, tm_pt, epochs=1000, verbose=False)
    print("Modelo entrenado!")

    print("Prediccion")
    r = modelo.predict(opp_av)
    print("Esperado " + str(r))
    return r




if b:

    Bt_ou_line = float(Bt_ou_line)

    urltm_H = ('https://www.basketball-reference.com/teams/' + pick_tm(tm_H[0]) + '/2022_games.html')

    df_i_H = pd.read_html(urltm_H, header=0)
    df_i_H = df_i_H[0]

    urltm_A = ('https://www.basketball-reference.com/teams/' + pick_tm(tm_A[0]) + '/2022_games.html')
    df_i_A = pd.read_html(urltm_A, header=0)
    df_i_A = df_i_A[0]

    urlT_defc = 'https://www.teamrankings.com/nba/stat/opponent-points-per-game'
    Def_pg = pd.read_html(urlT_defc)
    Def_pg = Def_pg[0]

    df_i_A = df_i_A.dropna(axis=0, subset=['Unnamed: 4'])
    df_i_H = df_i_H.dropna(axis=0, subset=['Unnamed: 4'])

    df_A = df_i_A.iloc[-+Last_G:]
    df_H = df_i_H.iloc[-+Last_G:]
    df_A = df_A.fillna(0, inplace=False)
    df_H = df_H.fillna(0, inplace=False)

    A_pt = df_A["Tm"]
    H_pt = df_H["Tm"]
    A_nop = df_A["Opponent"]
    H_nop = df_H["Opponent"]
    A_at = df_A["Unnamed: 5"]
    H_at = df_H["Unnamed: 5"]
    A_oppt = df_A["Opp"]
    H_oppt = df_H["Opp"]
    A_wl = df_A["Unnamed: 7"]
    H_wl = df_H["Unnamed: 7"]
    A_g = df_A["G"]
    H_g = df_H["G"]
    A_pt = np.asarray(A_pt)
    H_pt = np.asarray(H_pt)
    A_nop = np.asarray(A_nop)
    H_nop = np.asarray(H_nop)
    A_at = np.asarray(A_at)
    H_at = np.asarray(H_at)
    A_oppt = np.asarray(A_oppt)
    H_oppt = np.asarray(H_oppt)
    A_wl = np.asarray(A_wl)
    H_wl = np.asarray(H_wl)
    A_g = np.asarray(A_g)
    H_g = np.asarray(H_g)
    H_pt = H_pt.astype(np.int64)
    A_pt = A_pt.astype(np.int64)
    H_oppt = H_oppt.astype(np.int64)
    A_oppt = A_oppt.astype(np.int64)
    tot_H = np.add(H_pt,H_oppt)
    tot_A = np.add(A_pt, A_oppt)

    A_Def = np.zeros(0)
    H_Def = np.zeros(0)

    for h_nmi, ht in zip(H_nop, H_at):

        h_nm = tmnm(h_nmi)

        if ht == "@":
            loc = 5
        else:
            loc = 6

        H_pos = np.where(Def_pg == h_nm)
        Loc_def_H = Def_pg.iloc[H_pos[0], [loc]]
        Loc_def_H = np.asarray(Loc_def_H)
        H_Def = np.append(H_Def, [Loc_def_H])

    print(H_Def)

    for a_nmi, at in zip(A_nop, A_at):

        a_nm = tmnm(a_nmi)

        if at == "@":
            loc = 5
        else:
            loc = 6

        A_pos = np.where(Def_pg == a_nm)
        Loc_def_A = Def_pg.iloc[A_pos[0], [loc]]
        Loc_def_A = np.asarray(Loc_def_A)
        A_Def = np.append(A_Def, [Loc_def_A])

    print(A_Def)

    Opp_posH = np.where(Def_pg == tmnm(tm_A[0]))
    Opp_H = Def_pg.iloc[Opp_posH[0], [6]]
    Opp_H = Opp_H.astype(np.float64)

    Opp_posA = np.where(Def_pg == tmnm(tm_H[0]))
    Opp_A = Def_pg.iloc[Opp_posA[0], [5]]
    print(Opp_A)

    H_pt = H_pt.astype(np.int64)
    A_pt = A_pt.astype(np.int64)
    
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import lxml
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

logo_let = Image.open('data//C-La.png')
logo_ic = Image.open('data//C-L5.png')
st.set_page_config(page_title="ClutchWare-Player",page_icon=logo_ic)
Select_Pos=['PG','SG','SF','PF','C']
Select_Stat=['PTS','TRB','AST','3P','STL','BLK','TOV']
Write_Player = ""
Select_Last_Games = [7, 10]
Write_Bt_Line=0
Select_OPP=["Atlanta Hawks","Brooklyn Nets","Boston Celtics","Charlotte Hornets","Chicago Bulls","Cleveland Cavaliers","Dallas Mavericks","Denver Nuggets",
        "Detroit Pistons","Golden State Warriors","Houston Rockets","Indiana Pacers","Los Angeles Clippers","Los Angeles Lakers","Memphis Grizzlies","Miami Heat",
        "Milwaukee Bucks","Minnesota Timberwolves","New Orleans Pelicans","New York Knicks","Oklahoma City Thunder","Orlando Magic","Philadelphia 76ers","Phoenix Suns",
        "Portland Trail Blazers","Sacramento Kings","San Antonio Spurs","Toronto Raptors","Utah Jazz","Washington Wizards"]


st.image(logo_let)

st.header("Player Analyzer")
st.markdown("Artificial Intelligence NBA Player Prop Predictor")



Player_nm=st.sidebar.text_input("Player Name", Write_Player)
Opp_Percentage=st.sidebar.multiselect('VS.',Select_OPP)
Pos=st.sidebar.multiselect('Position',Select_Pos)
Prop_in=st.sidebar.multiselect('Prop',Select_Stat)
Last_n_Games_b=st.sidebar.multiselect('Last Games',Select_Last_Games)
Bt_line=st.sidebar.text_input("Bet Line")
Last_n_Games=np.dot(Last_n_Games_b, -1)
def Extraer_nombre(cadena, n1=5, n2=2):
    esp = Player_nm.index(' ')
    return cadena[esp + 1:esp + n1 + 1] + cadena[:n2]
def Extraer_inicial(cadena, n1=1):
    esp1 = Player_nm.index(' ')
    return cadena[esp1 + 1:esp1 + n1 + 1]
Player_nm_l = Player_nm.lower()

a=st.sidebar.button("Analyze")
if a:
        try:
            urlpla = ('https://www.basketball-reference.com/players/' + Extraer_inicial(Player_nm_l) + '/' + Extraer_nombre(
            Player_nm_l) + '01/gamelog/2022')
            print(urlpla)
            df = pd.read_html(urlpla, header=0)
        except ValueError:
            urlpla = ('https://www.basketball-reference.com/players/' + Extraer_inicial(Player_nm_l) + '/' + Extraer_nombre(
            Player_nm_l) + '02/gamelog/2022')
            df = pd.read_html(urlpla, header=0)


        if Pos[0]=='PG':
            urldef = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vT66hjhOQ8WyIQL67ihWKOcukBbwPVUUAt8cvYqTkGbnZTgo4XNPtgIknKyleZBL9O_KatA05BJECBl/pub?gid=0&single=true&output=csv'
        elif Pos[0]=='SG':
            urldef = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vT66hjhOQ8WyIQL67ihWKOcukBbwPVUUAt8cvYqTkGbnZTgo4XNPtgIknKyleZBL9O_KatA05BJECBl/pub?gid=503690944&single=true&output=csv'
        elif Pos[0]== 'SF':
            urldef ='https://docs.google.com/spreadsheets/d/e/2PACX-1vT66hjhOQ8WyIQL67ihWKOcukBbwPVUUAt8cvYqTkGbnZTgo4XNPtgIknKyleZBL9O_KatA05BJECBl/pub?gid=1088550471&single=true&output=csv'
        elif Pos[0] == 'PF':
            urldef ='https://docs.google.com/spreadsheets/d/e/2PACX-1vT66hjhOQ8WyIQL67ihWKOcukBbwPVUUAt8cvYqTkGbnZTgo4XNPtgIknKyleZBL9O_KatA05BJECBl/pub?gid=1638733906&single=true&output=csv'
        elif Pos[0] == 'C':
            urldef ='https://docs.google.com/spreadsheets/d/e/2PACX-1vT66hjhOQ8WyIQL67ihWKOcukBbwPVUUAt8cvYqTkGbnZTgo4XNPtgIknKyleZBL9O_KatA05BJECBl/pub?gid=831757291&single=true&output=csv'

        Pos_vs_def = pd.read_csv(urldef)
        len(df)
        dftb = df[7]

        try:
            dfplayer = dftb.drop(dftb.index [[20,41]])
        except IndexError:
            dfplayer = dftb.drop(dftb.index [[20]])


        if Prop_in[0] == "PTS":
            pr_ex = 1

        elif Prop_in[0] == "TRB":
            pr_ex = 2

        elif Prop_in[0] == "AST":
            pr_ex = 3

        elif Prop_in[0] == "3P":
            pr_ex = 4

        elif Prop_in[0] == "STL":
            pr_ex = 5

        elif Prop_in[0] == "BLK":
            pr_ex = 6

        elif Prop_in[0] == "TOV":
            pr_ex = 7


        Opp = dfplayer["Opp"].iloc[Last_n_Games[0]:]
        N_Ga = dfplayer["Rk"].iloc[Last_n_Games[0]:]
        PropPr = dfplayer[Prop_in].iloc[Last_n_Games[0]:]
        N_Ga_A = np.asarray(N_Ga)
        riv = np.asarray(Opp)
        Ar_Def = np.zeros(0)
        Ar_Player = np.zeros(0)

        for rival in riv:

            if rival == 'ATL':
                tm_ex = 0

            elif rival == "BRK":
                tm_ex = 1

            elif rival == "BOS":
                tm_ex = 2

            elif rival == "CHO":
                tm_ex = 3

            elif rival == "CHI":
                tm_ex = 4

            elif rival == "CLE":
                tm_ex = 5

            elif rival == "DAL":
                tm_ex = 6

            elif rival == "DEN":
                tm_ex = 7

            elif rival == "DET":
                tm_ex = 8

            elif rival == "GSW":
                tm_ex = 9

            elif rival == "HOU":
                tm_ex = 10

            elif rival == "IND":
                tm_ex = 11

            elif rival == "LAC":
                tm_ex = 12

            elif rival == "LAL":
                tm_ex = 13

            elif rival == "MEM":
                tm_ex = 14

            elif rival == "MIA":
                tm_ex = 15

            elif rival == "MIL":
                tm_ex = 16

            elif rival == "MIN":
                tm_ex = 17

            elif rival == "NOP":
                tm_ex = 18

            elif rival == "NYK":
                tm_ex = 19

            elif rival == "OKC":
                tm_ex = 20

            elif rival == "ORL":
                tm_ex = 21

            elif rival == "PHI":
                tm_ex = 22

            elif rival == "PHO":
                tm_ex = 23

            elif rival == "POR":
                tm_ex = 24

            elif rival == "SAC":
                tm_ex = 25

            elif rival == "SAS":
                tm_ex = 26

            elif rival == "TOR":
                tm_ex = 27

            elif rival == "UTA":
                tm_ex = 28

            elif rival == "WAS":
                tm_ex = 29

            Loc_def = Pos_vs_def.iloc[[tm_ex], [pr_ex]]
            Loc_def_Ar = np.asarray(Loc_def)
            Ar_Def = np.append(Ar_Def, [Loc_def_Ar])

        Ar_Player=np.asarray(PropPr)

        Ar_num = np.where(Ar_Player == "Inactive")
        Ar_num= np.asarray(Ar_num)
        Ar_num1 = np.where(Ar_Player == "Did Not Play")
        Ar_num1 = np.asarray(Ar_num1)

        for Ar_i in Ar_Player:

            if Ar_i == "Inactive" or Ar_i == "Did Not Play":
               Ar_Player[Ar_num[0]] = 0
               Ar_Def[Ar_num[0]] = 0
               Ar_Player[Ar_num1[0]] = 0
               Ar_Def[Ar_num1[0]] = 0


        if Opp_Percentage[0] == 'Atlanta Hawks':
            tm_ex = 0

        elif Opp_Percentage[0] == "Brooklyn Nets":
            tm_ex = 1

        elif Opp_Percentage[0] == "Boston Celtics":
            tm_ex = 2

        elif Opp_Percentage[0] == "Charlotte Hornets":
            tm_ex = 3

        elif Opp_Percentage[0]== "Chicago Bulls":
            tm_ex = 4

        elif Opp_Percentage[0] == "Cleveland Cavaliers":
            tm_ex = 5

        elif Opp_Percentage[0] == "Dallas Mavericks":
            tm_ex = 6

        elif Opp_Percentage[0] == "Denver Nuggets":
            tm_ex = 7

        elif Opp_Percentage[0] == "Detroit Pistons":
            tm_ex = 8

        elif Opp_Percentage[0] == "Golden State Warriors":
            tm_ex = 9

        elif Opp_Percentage[0] == "Houston Rockets":
            tm_ex = 10

        elif Opp_Percentage[0] == "Indiana Pacers":
            tm_ex = 11

        elif Opp_Percentage[0] == "Los Angeles Clippers":
            tm_ex = 12

        elif Opp_Percentage[0] == "Los Angeles Lakers":
            tm_ex = 13

        elif Opp_Percentage[0] == "Memphis Grizzlies":
            tm_ex = 14

        elif Opp_Percentage[0] == "Miami Heat":
            tm_ex = 15

        elif Opp_Percentage[0] == "Milwaukee Bucks":
            tm_ex = 16

        elif Opp_Percentage[0] == "Minnesota Timberwolves":
            tm_ex = 17

        elif Opp_Percentage[0] == "New Orleans Pelicans":
            tm_ex = 18

        elif Opp_Percentage[0] == "New York Knicks":
            tm_ex = 19

        elif Opp_Percentage[0] == "Oklahoma City Thunder":
            tm_ex = 20

        elif Opp_Percentage[0] == "Orlando Magic":
            tm_ex = 21

        elif Opp_Percentage[0] == "Philadelphia 76ers":
            tm_ex = 22

        elif Opp_Percentage[0] == "Phoenix Suns":
            tm_ex = 23

        elif Opp_Percentage[0] == "Portland Trail Blazers":
            tm_ex = 24

        elif Opp_Percentage[0] == "Sacramento Kings":
            tm_ex = 25

        elif Opp_Percentage[0] == "San Antonio Spurs":
            tm_ex = 26

        elif Opp_Percentage[0] == "Toronto Raptors":
            tm_ex = 27

        elif Opp_Percentage[0] == "Utah Jazz":
            tm_ex = 28

        elif Opp_Percentage[0] == "Washington Wizards":
            tm_ex = 29

        Loc_def_P = Pos_vs_def.iloc[[tm_ex], [pr_ex]]
        Opp_P_r = Loc_def_P.astype(np.float64)
        Opp_plt= np.asarray(Opp_P_r)

        Player = Ar_Player.astype(np.int64)
        Defense = Ar_Def.astype(np.float64)


        Player_df=pd.DataFrame(Player, columns = ["Player"])
        Defense_nm_df = pd.DataFrame(riv, columns=["Team"])
        N_Ga_df = pd.DataFrame(N_Ga_A, columns=["N°"])
        Defense_df = pd.DataFrame(Defense, columns=["Defense Avg."])
        Tabla_chido = pd.concat([Player_df, Defense_nm_df,Defense_df,N_Ga_df],axis=1)
        Opp_P = pd.DataFrame(Opp_plt, columns=["%"])





        capa = tf.keras.layers.Dense(units=1, input_shape=[1])
        modelo = tf.keras.Sequential([capa])

        oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
        oculta2 = tf.keras.layers.Dense(units=3)
        salida = tf.keras.layers.Dense(units=1)
        modelo = tf.keras.Sequential([oculta1, oculta2, salida])
        modelo.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='mean_squared_error'
        )





    
    with st.spinner("Starting training..."):
             
      rH = AI(H_pt, H_Def, Opp_H)

      rA = AI(A_pt, A_Def, Opp_A)
    st.success("Model trained!")
    res = rH + rA
    resS = rH + rA + 3.5


    if res > Bt_ou_line:
        line = Bt_ou_line - 4.5
        t_line = str('Over ' + str(line))

    else:
        line = Bt_ou_line + 4.5
        t_line = str('Under ' + str(line))

    if rH > rA:

        Ml = str(tm_H[0] + " is expected to win")

    else:

        Ml = str(tm_A[0] + " is expected to win")

    tot_df_H=pd.DataFrame(tot_H,columns=["Total Home"])
    pt_df_H=pd.DataFrame(H_pt,columns=["Tm"])
    oppt_df_H = pd.DataFrame(H_oppt, columns=["Opp"])
    g_df_H = pd.DataFrame(H_g, columns = ["N°"])
    H_df_pl = pd.concat([tot_df_H,g_df_H,pt_df_H,oppt_df_H],axis=1)

    tot_df_A = pd.DataFrame(tot_A, columns=["Total Away"])
    pt_df_A=pd.DataFrame(A_pt,columns=["Tm"])
    oppt_df_A = pd.DataFrame(A_oppt, columns=["Opp"])
    g_df_A = pd.DataFrame(A_g, columns = ["N°"])
    A_df_pl = pd.concat([tot_df_A, g_df_A,pt_df_A,oppt_df_A], axis=1)




    tot_Graph_H = px.bar(H_df_pl, x="N°", y="Total Home",
                 color_discrete_sequence=['#66FCF1', '#45A29E'],
                 title=tm_H[0]+" last total points")
    tot_Graph_H.add_hline(y=int(res), line_color="#00ECFF", line_width=4)
    for i, t in enumerate([H_nop]):
        tot_Graph_H.data[i].text = t
        tot_Graph_H.data[i].textposition = 'inside'

    tot_Graph_A = px.bar(A_df_pl, x="N°", y="Total Away",
                 color_discrete_sequence=['#66FCF1', '#45A29E'],
                 title=tm_A[0]+" last total points")
    tot_Graph_A.add_hline(y=int(res), line_color="#00ECFF", line_width=4)
    for i, t in enumerate([A_nop]):
        tot_Graph_A.data[i].text = t
        tot_Graph_A.data[i].textposition = 'inside'

    fig = px.bar(H_df_pl, x='N°', y=["Tm", "Opp"],
                 color_discrete_sequence=['#66FCF1', '#45A29E'] * len(H_df_pl), barmode='group',
                 title=tm_H[0]+" Vs")
    for i, t in enumerate([" ",H_nop]):
        fig.data[i].text = t
        fig.data[i].textposition = 'inside'


    fig2 = px.bar(A_df_pl, x='N°', y=["Tm", "Opp"],
                 color_discrete_sequence=['#66FCF1', '#45A29E'] * len(A_df_pl), barmode='group',
                 title=tm_A[0]+" Vs")
    for i, t in enumerate([" ",A_nop]):
        fig2.data[i].text = t
        fig2.data[i].textposition = 'inside'

    m2, m3, m4 = st.columns(3)
    m2.metric(label=tm_H[0], value=str(np.round(rH[0],3)) , delta="Expected Points", delta_color="off")
    m3.metric(label=tm_A[0], value=str(np.round(rA[0], 3)), delta="Expected Points", delta_color="off")
    m4.metric(label="Total", value=str(np.round(res[0], 3)), delta="Expected Points", delta_color="off")
    st.info("Recommended total line -"+t_line)
    st.info(Ml)

    st.plotly_chart(tot_Graph_H)
    st.plotly_chart(tot_Graph_A)
    st.plotly_chart(fig)
    st.plotly_chart(fig2)

    st.write(urltm_H)
    st.write(urltm_A)



