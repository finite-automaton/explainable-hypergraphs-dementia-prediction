import pandas as pd
import random

TWO_VALS = ["NACCNREX","STEPWISE","SOMATIC","EMOT","HXHYPER","PDOTHR","NACCAHTN","NACCHTNC","NACCACEI","NACCAAAS","NACCBETA","NACCCCBS","NACCDIUR","NACCVASD","NACCANGI","NACCLIPL","NACCNSD","NACCAC","NACCADEP","NACCAPSY","NACCAANX","NACCADMD","NACCPDMD","NACCEMD","NACCEPMD","NACCDBMD","DECCLIN","EYEMOVE","FOCLDEF","GAITDIS","EMPTY","DROPACT","BORED","HELPLESS","WRTHLESS","HOPELESS","AFRAID","BETTER","STAYHOME","MEMPROB","TOBAC30","DEP2YRS","DEPOTHR","NACCTBI","TOBAC100","DEL","HALL","AGIT","ELAT","DISN","DEPD","APA","MOT","ANX","IRR","APP","NITE","STROKE","HEARAID","VISCORR","HEARING","VISION","NACCFAM","NACCDAD","NACCMOM","PDNORMAL","SPIRITS","SATIS","ENERGY","WONDRFUL","HAPPY","DECIN","MOFALLS","BEDEP","BEAHALL","BEDEL","COGATTN","COGVIS","BEPERCH","MOGAIT","COGJUDG","BEVHALL","COGOTHR","COGLANG","MOTREM","BEDISIN","BEAPATHY","MOSLOW","BEIRRIT","BEAGIT","NACCFADM","NACCFFTD","PARK","BRNINJ","HYCEPH","DEP","OTHPSY","NACCADMU","NACCFTDM"]
TWO_VALSX = ["ABRUPT", "HXSTROKE", "FOCLSYM", "FOCLSIGN"]
THREE_VALS = ["HYPERTEN","THYROID","HYPERCHO","B12DEF","TRAUMCHR","TRAUMEXT","NCOTHR","TRAUMBRF","ABUSOTHR","PSYCDIS","ALCOHOL","CBSTROKE","CVAFIB","SEIZURES","CVOTHR","CBTIA","CVPACE","INCONTF","CVBYPASS","INCONTU","CVHATT","CVANGIO","DIABETES","CVCHF"]
THREE_VALSX = ["DECSUB", "VISWCORR","HEARWAID"]
FOUR_VALSX = ["DIABTYPE", "FRSTCHG"]
FOUR_VALSY = ["INDEPEND", "RESIDENC"]
FIVE_VALS = ["BEMODE", "COGMODE", "NACCMOTF", "MOMODE"]
FIVE_VALSX = ["SHOPPING","TRAVEL","EVENTS","STOVE","REMDATES","MEALPREP","BILLS","PAYATTN","TAXES","GAMES"]
FIVE_VALSY = ["NACCLIVS"]
SIX_VALSX = ["BRADYKIN","TAPSRT","TAPSLF","HANDMOVR","TRESTFAC","TRESTRHD","TRESTLHD","TRESTRFT","TRESTLFT","RIGDNECK","RIGDUPRT","RIGDLORT","RIGDLOLF","TRACTRHD","TRACTLHD","HANDMOVL","HANDALTR","HANDALTL","LEGRT","LEGLF","FACEXP","POSTURE","SPEECH","ARISING","GAIT","POSSTAB","RIGDUPLF"]
SIX_VALSXY = ["COURSE"]
SIX_VALSY = ["MARISTAT", "NACCNIHR"]
SEVEN_VALSX = ["PACKSPER"]
NINE_VALS = ["NACCCOGF"]
ELEVEN_VALS = ["NACCBEHF"]
THIRTEEN_VALS = ["HACHIN"]

def generate_synthetic_data(num_rows):
    # Define the value ranges for each field
    HG_TWO_VALS = [0, 1]
    HG_TWO_VALSX = [0, 2]
    HG_THREE_VALS = [0, 1, 2]
    HG_THREE_VALSX = [0, 1, 8]
    HG_FOUR_VALSX = [1, 2, 3, 8]
    HG_FOUR_VALSY = [1, 2, 3, 4]
    HG_FIVE_VALS = [0, 1, 2, 3, 4]
    HG_FIVE_VALSX = [0, 1, 2, 3, 8, 9]
    HG_FIVE_VALSY = [1, 2, 3, 4, 5, 9]
    HG_SIX_VALSX = [0, 1, 2, 3, 4, 8]
    HG_SIX_VALSXY = [1, 2, 3, 4, 5, 8, 9]
    HG_SIX_VALSY = [1, 2, 3, 4, 5, 6]
    HG_SEVEN_VALSX = [0, 1, 2, 3, 4, 5, 8, 9, -4]
    HG_NINE_VALS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    HG_ELEVEN_VALS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 99]
    HG_THIRTEEN_VALS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -4]

    def random_numeric(min_val, max_val, special_values, prob_special=0.05):
        if special_values and random.random() < prob_special:
            return random.choice(special_values)
        else:
            return random.uniform(min_val, max_val) if isinstance(min_val, float) else random.randint(min_val, max_val)

    data = {
        'NACCID': [], 'NACCVNUM': [], 'NACCAVST': [],
        "NACCNREX": [], "STEPWISE": [], "SOMATIC": [], "EMOT": [], "HXHYPER": [], "PDOTHR": [],
        "NACCAHTN": [], "NACCHTNC": [], "NACCACEI": [], "NACCAAAS": [], "NACCBETA": [], "NACCCCBS": [],
        "NACCDIUR": [], "NACCVASD": [], "NACCANGI": [], "NACCLIPL": [], "NACCNSD": [], "NACCAC": [],
        "NACCADEP": [], "NACCAPSY": [], "NACCAANX": [], "NACCADMD": [], "NACCPDMD": [], "NACCEMD": [],
        "NACCEPMD": [], "NACCDBMD": [], "DECCLIN": [], "EYEMOVE": [], "FOCLDEF": [], "GAITDIS": [],
        "EMPTY": [], "DROPACT": [], "BORED": [], "HELPLESS": [], "WRTHLESS": [], "HOPELESS": [],
        "AFRAID": [], "BETTER": [], "STAYHOME": [], "MEMPROB": [], "TOBAC30": [], "DEP2YRS": [],
        "DEPOTHR": [], "NACCTBI": [], "TOBAC100": [], "DEL": [], "HALL": [], "AGIT": [], "ELAT": [],
        "DISN": [], "DEPD": [], "APA": [], "MOT": [], "ANX": [], "IRR": [], "APP": [], "NITE": [],
        "STROKE": [], "HEARAID": [], "VISCORR": [], "HEARING": [], "VISION": [], "NACCFAM": [],
        "NACCDAD": [], "NACCMOM": [], "PDNORMAL": [], "SPIRITS": [], "SATIS": [], "ENERGY": [],
        "WONDRFUL": [], "HAPPY": [], "DECIN": [], "MOFALLS": [], "BEDEP": [], "BEAHALL": [], "BEDEL": [],
        "COGATTN": [], "COGVIS": [], "BEPERCH": [], "MOGAIT": [], "COGJUDG": [], "BEVHALL": [],
        "COGOTHR": [], "COGLANG": [], "MOTREM": [], "BEDISIN": [], "BEAPATHY": [], "MOSLOW": [],
        "BEIRRIT": [], "BEAGIT": [], "NACCFADM": [], "NACCFFTD": [], "PARK": [], "BRNINJ": [],
        "HYCEPH": [], "DEP": [], "OTHPSY": [], "NACCADMU": [], "NACCFTDM": [],
        "ABRUPT": [], "HXSTROKE": [], "FOCLSYM": [], "FOCLSIGN": [],
        "HYPERTEN": [], "THYROID": [], "HYPERCHO": [], "B12DEF": [], "TRAUMCHR": [], "TRAUMEXT": [],
        "NCOTHR": [], "TRAUMBRF": [], "ABUSOTHR": [], "PSYCDIS": [], "ALCOHOL": [], "CBSTROKE": [],
        "CVAFIB": [], "SEIZURES": [], "CVOTHR": [], "CBTIA": [], "CVPACE": [], "INCONTF": [],
        "CVBYPASS": [], "INCONTU": [], "CVHATT": [], "CVANGIO": [], "DIABETES": [], "CVCHF": [],
        "DECSUB": [], "VISWCORR": [], "HEARWAID": [],
        "DIABTYPE": [], "FRSTCHG": [],
        "INDEPEND": [], "RESIDENC": [],
        "BEMODE": [], "COGMODE": [], "NACCMOTF": [], "MOMODE": [],
        "SHOPPING": [], "TRAVEL": [], "EVENTS": [], "STOVE": [], "REMDATES": [], "MEALPREP": [],
        "BILLS": [], "PAYATTN": [], "TAXES": [], "GAMES": [],
        "NACCLIVS": [],
        "BRADYKIN": [], "TAPSRT": [], "TAPSLF": [], "HANDMOVR": [], "TRESTFAC": [], "TRESTRHD": [],
        "TRESTLHD": [], "TRESTRFT": [], "TRESTLFT": [], "RIGDNECK": [], "RIGDUPRT": [], "RIGDLORT": [],
        "RIGDLOLF": [], "TRACTRHD": [], "TRACTLHD": [], "HANDMOVL": [], "HANDALTR": [], "HANDALTL": [],
        "LEGRT": [], "LEGLF": [], "FACEXP": [], "POSTURE": [], "SPEECH": [], "ARISING": [],
        "GAIT": [], "POSSTAB": [], "RIGDUPLF": [],
        "COURSE": [],
        "MARISTAT": [], "NACCNIHR": [],
        "PACKSPER": [],
        "NACCCOGF": [],
        "NACCBEHF": [],
        "HACHIN": [],
        'NACCBMI': [], 'BPDIAS': [], 'BPSYS': [], 'HANDED': [], 'EDUC': [], 'NACCAGE': [],
        'NACCAPOE': [], 'NACCNE4S': [], 'SEX': [], 'MEMORY': [], 'ORIENT': [], 'JUDGMENT': [],
        'COMMUN': [], 'HOMEHOBB': [], 'PERSCARE': [], 'INRELY': [], 'CDRGLOB': []
    }

    # Generate data
    naccid = 1
    while len(data['NACCID']) < num_rows:
        num_visits = random.randint(1, 12)  # Each patient can have up to 12 visits
        progressed = random.choice([0, 1])
        for visit in range(1, num_visits + 1):
            if len(data['NACCID']) >= num_rows:
                break
            data['NACCID'].append(naccid)
            data['NACCVNUM'].append(visit)
            data['NACCAVST'].append(num_visits)
            for code in TWO_VALS:
                data[code].append(random.choice(HG_TWO_VALS))
            for code in TWO_VALSX:
                data[code].append(random.choice(HG_TWO_VALSX))
            for code in THREE_VALS:
                data[code].append(random.choice(HG_THREE_VALS))
            for code in THREE_VALSX:
                data[code].append(random.choice(HG_THREE_VALSX))
            for code in FOUR_VALSX:
                data[code].append(random.choice(HG_FOUR_VALSX))
            for code in FOUR_VALSY:
                data[code].append(random.choice(HG_FOUR_VALSY))
            for code in FIVE_VALS:
                data[code].append(random.choice(HG_FIVE_VALS))
            for code in FIVE_VALSX:
                data[code].append(random.choice(HG_FIVE_VALSX))
            for code in FIVE_VALSY:
                data[code].append(random.choice(HG_FIVE_VALSY))
            for code in SIX_VALSX:
                data[code].append(random.choice(HG_SIX_VALSX))
            for code in SIX_VALSXY:
                data[code].append(random.choice(HG_SIX_VALSXY))
            for code in SIX_VALSY:
                data[code].append(random.choice(HG_SIX_VALSY))
            for code in SEVEN_VALSX:
                data[code].append(random.choice(HG_SEVEN_VALSX))
            for code in NINE_VALS:
                data[code].append(random.choice(HG_NINE_VALS))
            for code in ELEVEN_VALS:
                data[code].append(random.choice(HG_ELEVEN_VALS))
            for code in THIRTEEN_VALS:
                data[code].append(random.choice(HG_THIRTEEN_VALS))
            data['NACCBMI'].append(random_numeric(10.0, 100.0, [888.8, -4]))
            data['BPDIAS'].append(random_numeric(30, 140, [888, -4]))
            data['BPSYS'].append(random_numeric(70, 230, [888, -4]))
            data['HANDED'].append(random.choice([1, 2, 3, 9]))
            data['EDUC'].append(random_numeric(0, 36, [99]))
            data['NACCAGE'].append(random_numeric(50, 90, []))
            data['NACCAPOE'].append(random.choice([1, 2, 3, 4, 5, 6]))
            data['NACCNE4S'].append(random.choice([0, 1, 2]))
            data['SEX'].append(random.choice([1, 2]))
            data['MEMORY'].append(random.choice([0.0, 0.5, 1.0, 2.0, 3.0]))
            data['ORIENT'].append(random.choice([0.0, 0.5, 1.0, 2.0, 3.0]))
            data['JUDGMENT'].append(random.choice([0.0, 0.5, 1.0, 2.0, 3.0]))
            data['COMMUN'].append(random.choice([0.0, 0.5, 1.0, 2.0, 3.0]))
            data['HOMEHOBB'].append(random.choice([0.0, 0.5, 1.0, 2.0, 3.0]))
            data['PERSCARE'].append(random.choice([0.0, 0.5, 1.0, 2.0, 3.0]))
            data['INRELY'].append(0)
            if progressed == 0:
                data['CDRGLOB'].append(0.5)
            elif progressed == 1 and visit == 1:
                data['CDRGLOB'].append(0.5)
            else:
                data['CDRGLOB'].append(1)
        naccid += 1



    # Create DataFrame
    df = pd.DataFrame(data)

    # Export to CSV
    df.to_csv('hypergraph_data/raw_data/raw_data.csv', index=False)

    return df

# change this value only to change the number of synthetic rows, a value which is too low may
# not produce enough 'valid' rows for the model
NUM_ROWS = 10000
df = generate_synthetic_data(NUM_ROWS)
