#!/usr/bin/env python3
"""The place-name gazetteer: every place the analysis counts, classified by hand.

Built from the candidate list written by place_candidates.py (every string a
named-entity model took for a place, plus every capitalised string occurring
three times or more). Each line below is

    canonical name | kind | area | spellings, separated by semicolons

kind   ua       a settlement or region inside Ukraine's 1991 borders
       foreign  a settlement or region outside them
       state    a sovereign state, empire or continent (counted separately)
area   for `ua`: macro-region West / Centre / South / East, by oblast --
           West:   Volyn, Rivne, Lviv, Ivano-Frankivsk, Ternopil, Zakarpattia,
                   Khmelnytskyi, Chernivtsi
           Centre: Kyiv and its oblast, Vinnytsia, Zhytomyr, Sumy, Chernihiv,
                   Poltava, Kirovohrad, Cherkasy
           South:  Dnipropetrovsk, Zaporizhzhia, Mykolaiv, Kherson, Odesa, Crimea
           East:   Kharkiv, Donetsk, Luhansk
       for `foreign` and `state`: the present-day country (or "historical")

Spellings cover the English and German transliterations used in the five books,
the older Russian- and Polish-based forms, and the misreadings the scans
produce ("Kénigsberg", "Lw6w"). Matching is case-sensitive and allows a
possessive or genitive ending. Seas, rivers and streets are not counted.

Deliberately left out because the string is ambiguous in these books:
"New York" (the city and the Donetsk-oblast town), "Polesia" (straddles the
border), "Maidan" (a square and a movement), "Galicia" is kept as western
Ukraine, which is the sense all five books use.
"""
from __future__ import annotations

import re

RAW = """
# ── Ukraine: West ──
Lviv | ua | West | Lviv; L'viv; L’viv; Lvov; Lwów; Lwow; Lw6w; Lwéw; Lemberg; Lwiw; Lvív
Ivano-Frankivsk | ua | West | Ivano-Frankivsk; Iwano-Frankiwsk; Ivano-Frankovsk; Stanislau; Stanislav; Stanisławów
Ternopil | ua | West | Ternopil; Tarnopol
Lutsk | ua | West | Lutsk; Luzk; Łuck
Rivne | ua | West | Rivne; Riwne; Rovno
Uzhhorod | ua | West | Uzhhorod; Uschhorod; Uzhgorod; Ungvár
Mukachevo | ua | West | Mukachevo; Mukatschewo; Munkács
Chernivtsi | ua | West | Chernivtsi; Czernowitz; Tscherniwzi; Tschwerniwzi; Chernovtsy; Cernauti; Cernăuți; Cernauri; Chernivtsy
Khmelnytskyi | ua | West | Khmelnytskyi; Chmelnyzkyj; Khmelnitsky
Kamianets-Podilskyi | ua | West | Kamenets Podolsky; Kamianets-Podilskyi; Kamenets-Podolsky; Kamieniec Podolski; Kamjanez-Podilskyj; Kamenets; Kamieniec
Drohobych | ua | West | Drohobych; Drohobytsch; Drohobycz
Truskavets | ua | West | Truskavets; Truskawez
Kalush | ua | West | Kalush; Kalusch
Kolomyia | ua | West | Kolomyia; Kolomyja; Kolomea
Woroniaki | ua | West | Woroniaki; Voroniaky
Brody | ua | West | Brody
Zolochiv | ua | West | Zolochiv; Złoczów; Zloczow
Stryi | ua | West | Stryi; Stryj
Yaremche | ua | West | Yaremche; Jaremtsche
Bukovel | ua | West | Bukovel; Bukowel
Khotyn | ua | West | Khotyn; Chotyn; Chocim
Rakhiv | ua | West | Rakhiv; Rachiw
Vorokhta | ua | West | Vorokhta; Worochta
Verkhovyna | ua | West | Verkhovyna; Werchowyna
Kosiv | ua | West | Kosiv; Kossiw
Duliby | ua | West | Duliby
Khust | ua | West | Khust; Chust
Krasne | ua | West | Krasne
Unizh | ua | West | Unizh
Yavoriv | ua | West | Yavoriv; Jaworiw
Mostyska | ua | West | Mostyska
Dilove | ua | West | Dilove; Dilowe
Western Ukraine | ua | West | Westukraine
Galicia | ua | West | Galicia; Galizien; Halychyna
Bukovyna | ua | West | Bukovyna; Bukovina; Bukowina
Transcarpathia | ua | West | Transcarpathia; Transkarpatien; Zakarpattia; Sakarpattja; Ruthenia; Subcarpathian Ruthenia
Volhynia | ua | West | Volhynia; Wolhynien; Volyn; Wolyn
Podolia | ua | West | Podolia; Podolien; Podillia
Carpathians | ua | West | Carpathians; Carpathian Mountains; Karpaten
# ── Ukraine: Centre ──
Kyiv | ua | Centre | Kyiv; Kiev; Kyjiw; Kiew; Kijew
Vinnytsia | ua | Centre | Vinnytsia; Winnyzja; Vinnitsa; Vinnytsya
Zhytomyr | ua | Centre | Zhytomyr; Schytomyr; Zhitomir; Zhytomir
Cherkasy | ua | Centre | Cherkasy; Tscherkassy; Tscherkasy; Cherkassy
Kropyvnytskyi | ua | Centre | Kropyvnytskyi; Kirovohrad; Kropywnyzkyj; Kirowohrad
Poltava | ua | Centre | Poltava; Poltawa
Chernihiv | ua | Centre | Chernihiv; Tschernihiw; Tschernihiv; Chernigov
Sumy | ua | Centre | Sumy
Bila Tserkva | ua | Centre | Bila Tserkva; Bila Zerkwa
Irpin | ua | Centre | Irpin
Bucha | ua | Centre | Bucha; Butscha
Hostomel | ua | Centre | Hostomel
Borodianka | ua | Centre | Borodianka; Borodjanka; Borodyanka
Vasylkiv | ua | Centre | Vasylkiv; Wassylkiw
Brovary | ua | Centre | Brovary; Browary
Makariv | ua | Centre | Makariv; Makariw
Moshchun | ua | Centre | Moshchun; Moschtschun
Chornobyl | ua | Centre | Chornobyl; Chernobyl; Tschernobyl; Tschornobyl
Prypiat | ua | Centre | Prypiat; Pripyat; Prypjat
Uman | ua | Centre | Uman
Kremenchuk | ua | Centre | Kremenchuk; Krementschuk
Myrhorod | ua | Centre | Myrhorod
Lukashivka | ua | Centre | Lukashivka; Lukaschiwka
Yahidne | ua | Centre | Yahidne; Jahidne
Berdychiv | ua | Centre | Berdychiv; Berdytschiw; Berdichev
Kaniv | ua | Centre | Kaniv; Kaniw
Okhtyrka | ua | Centre | Okhtyrka; Ochtyrka
Konotop | ua | Centre | Konotop
Nizhyn | ua | Centre | Nizhyn; Nischyn
Babyn Yar | ua | Centre | Babyn Yar; Babi Yar; Babyn Jar
Podil | ua | Centre | Podil
Mezhyhirya | ua | Centre | Mezhyhirya; Meschyhirja
Boryspil | ua | Centre | Boryspil
Ivankiv | ua | Centre | Ivankiv; Iwankiw
Lokhvytsia | ua | Centre | Lokhvytsia; Lochwyzja
Opishnia | ua | Centre | Oposchnija; Opaschnija; Opishnia
Troieshchyna | ua | Centre | Trojeschina; Troieshchyna
Dykanka | ua | Centre | Dykanka; Dikanka
# ── Ukraine: South ──
Odesa | ua | South | Odesa; Odessa
Dnipro | ua | South | Dnipro; Dnipropetrovsk; Dnipropetrovs'k; Dnipropetrovs’k; Dnipropetrowsk; Dnepropetrovsk; Dnjepropetrowsk
Zaporizhzhia | ua | South | Zaporizhzhia; Saporischschja; Zaporozhye; Zaporizhia; Saporischja
Mykolaiv | ua | South | Mykolaiv; Mykolajiw; Mykolayiv; Nikolaev; Nikolayev
Kherson | ua | South | Kherson; Cherson
Crimea | ua | South | Crimea; Krim
Simferopol | ua | South | Simferopol
Belbek | ua | South | Belbek
Armiansk | ua | South | Armiansk
Yevpatoria | ua | South | Yevpatoria; Jewpatorija
Feodosia | ua | South | Feodosia; Feodossija
Skadovsk | ua | South | Skadovsk
Stepnohirsk | ua | South | Stepnohirsk
Robotyne | ua | South | Robotyne
Khortytsia | ua | South | Chortyzja; Khortytsia
Snake Island | ua | South | Snake Island; Schlangeninsel
Sevastopol | ua | South | Sevastopol; Sewastopol
Yalta | ua | South | Yalta; Jalta
Kerch | ua | South | Kerch; Kertsch
Bakhchysarai | ua | South | Bakhchysarai; Bachtschyssaraj; Bakhchisaray
Melitopol | ua | South | Melitopol
Berdiansk | ua | South | Berdiansk; Berdjansk; Berdyansk
Kryvyi Rih | ua | South | Kryvyi Rih; Krywyj Rih; Krivoy Rog
Enerhodar | ua | South | Enerhodar
Nikopol | ua | South | Nikopol
Kakhovka | ua | South | Kakhovka; Kachowka; Nova Kakhovka; Nowa Kachowka
Pavlohrad | ua | South | Pavlohrad; Pawlohrad
Kamianske | ua | South | Kamianske; Kamjanske; Dniprodzerzhynsk
Izmail | ua | South | Izmail; Ismajil
Orikhiv | ua | South | Orikhiv; Orichiw
Huliaipole | ua | South | Huliaipole; Huljajpole
Novorossiya | ua | South | Novorossiya; Noworossija
# ── Ukraine: East ──
Kharkiv | ua | East | Kharkiv; Charkiw; Kharkov; Charkow
Donetsk | ua | East | Donetsk; Donezk
Luhansk | ua | East | Luhansk; Lugansk
Donbas | ua | East | Donbas; Donbass
Mariupol | ua | East | Mariupol
Bakhmut | ua | East | Bakhmut; Bachmut; Artemivsk; Artemiwsk; Artemovsk; Artyomovsk
Kramatorsk | ua | East | Kramatorsk
Sloviansk | ua | East | Sloviansk; Slovyansk; Slowjansk; Slavyansk
Sievierodonetsk | ua | East | Sievierodonetsk; Severodonetsk; Sjewjerodonezk
Lysychansk | ua | East | Lysychansk; Lyssytschansk
Debaltseve | ua | East | Debaltseve; Debalzewe
Ilovaisk | ua | East | Ilovaisk; Ilowajsk
Avdiivka | ua | East | Avdiivka; Awdijiwka
Mariinka | ua | East | Mariinka; Marjinka; Marinka
Horlivka | ua | East | Horlivka; Horliwka; Gorlovka
Torez | ua | East | Torez; Tores
Izium | ua | East | Izium; Izyum; Isjum
Kupiansk | ua | East | Kupiansk; Kupjansk
Lyman | ua | East | Lyman
Kostiantynivka | ua | East | Kostiantynivka; Kostjantyniwka; Konstantinovka; Kostyantynivka
Pokrovsk | ua | East | Pokrovsk; Pokrowsk
Chasiv Yar | ua | East | Chasiv Yar; Tschassiw Jar
Soledar | ua | East | Soledar
Volnovakha | ua | East | Volnovakha; Wolnowacha
Saltivka | ua | East | Saltivka; Saltiwka
Lozova | ua | East | Lozova; Losowa
Blyzniuky | ua | East | Blyzniuky; Blysnjuky; Blysjuky
Balakliia | ua | East | Balakliia; Balaklija
Chuhuiv | ua | East | Chuhuiv; Tschuhujiw
Popasna | ua | East | Popasna
Shchastia | ua | East | Shchastia; Schtschastja
Hrabove | ua | East | Hrabove
Pervomaiske | ua | East | Pervomaiske
Marinovka | ua | East | Marinovka
Verkhnya Krynka | ua | East | Verkhnya Krynka
Snizhne | ua | East | Snizhne
Druzhkivka | ua | East | Druzhkivka; Druschkiwka
Toretsk | ua | East | Toretsk; Torezk
Vuhledar | ua | East | Vuhledar; Wuhledar
Yenakiieve | ua | East | Yenakiieve; Yenakiyevo
Ivanivske | ua | East | Ivanivske
Sviatohirsk | ua | East | Sviatohirsk; Swjatohirsk
Bohorodychne | ua | East | Bohorodychne; Bohorodytschne
Makiivka | ua | East | Makiivka; Makijiwka
Opytne | ua | East | Opytne
Raihorodok | ua | East | Raihorodok
Novoazovsk | ua | East | Novoazovsk
Karlivka | ua | East | Karlivka
Olenivka | ua | East | Olenivka
Khromove | ua | East | Khromove
Paraskoviivka | ua | East | Paraskoviivka
Velyka Novosilka | ua | East | Velyka Novosilka
Zhdanivka | ua | East | Zhdanivka
Kurakhove | ua | East | Kurakhove
Siedove | ua | East | Siedove; Sedove
Bilohorivka | ua | East | Bilohorivka
Mykolaivka | ua | East | Mykolaivka
Nyzhnya Krynka | ua | East | Nyzhnya Krynka
Bezimenne | ua | East | Bezimenne
Eastern Ukraine | ua | East | Ostukraine
# ── Foreign places: Russia ──
Moscow | foreign | Russia | Moscow; Moskau
Saint Petersburg | foreign | Russia | Saint Petersburg; St Petersburg; St. Petersburg; Sankt Petersburg; Petersburg; Leningrad
Kaliningrad | foreign | Russia | Kaliningrad; Königsberg; Konigsberg; Kénigsberg; K6nigsberg; Koenigsberg
Svetlogorsk | foreign | Russia | Svetlogorsk
Rostov-on-Don | foreign | Russia | Rostov-on-Don; Rostov; Rostow
Siberia | foreign | Russia | Siberia; Sibirien
Tomsk | foreign | Russia | Tomsk
Irkutsk | foreign | Russia | Irkutsk
Krasnoyarsk | foreign | Russia | Krasnoyarsk
Novosibirsk | foreign | Russia | Novosibirsk
Yekaterinburg | foreign | Russia | Yekaterinburg; Ekaterinburg; Sverdlovsk
Samara | foreign | Russia | Samara
Voronezh | foreign | Russia | Voronezh
Orenburg | foreign | Russia | Orenburg
Ufa | foreign | Russia | Ufa
Perm | foreign | Russia | Perm
Barnaul | foreign | Russia | Barnaul
Ulan-Ude | foreign | Russia | Ulan-Ude
Baikal | foreign | Russia | Baikal; Olkhon
Belgorod | foreign | Russia | Belgorod
Kursk | foreign | Russia | Kursk
Grozny | foreign | Russia | Grozny; Grosny
Chechnya | foreign | Russia | Chechnya; Tschetschenien
Volgograd | foreign | Russia | Volgograd; Wolgograd; Stalingrad
Saratov | foreign | Russia | Saratov; Saratow
Sochi | foreign | Russia | Sochi; Sotschi
Muscovy | foreign | Russia | Muscovy
Novgorod | foreign | Russia | Novgorod
Omsk | foreign | Russia | Omsk
Smolensk | foreign | Russia | Smolensk
Vladivostok | foreign | Russia | Vladivostok; Wladiwostok
Taganrog | foreign | Russia | Taganrog
Pyatigorsk | foreign | Russia | Pyatigorsk
Yakutsk | foreign | Russia | Jakutsk; Yakutsk
Urals | foreign | Russia | Urals; Ural
Caucasus | foreign | historical | Caucasus; Kaukasus
Central Asia | foreign | historical | Central Asia; Zentralasien
# ── Belarus ──
Minsk | foreign | Belarus | Minsk
Brest | foreign | Belarus | Brest
Kobrin | foreign | Belarus | Kobrin
Pinsk | foreign | Belarus | Pinsk
Nowogródek | foreign | Belarus | Nowogródek; Nowogrodek; Nowogrédek; Nowogr6dek; Novogrudok; Navahrudak
Radun | foreign | Belarus | Radun
Hermaniszki | foreign | Belarus | Hermaniszki
Bieniakonie | foreign | Belarus | Bieniakonie
Woronowa | foreign | Belarus | Woronowa
Grodno | foreign | Belarus | Grodno; Hrodna
Bolcieniki | foreign | Belarus | Bolcieniki
# ── Lithuania ──
Vilnius | foreign | Lithuania | Vilnius; Wilno; Vilna
Kaunas | foreign | Lithuania | Kaunas
Paberžė | foreign | Lithuania | Paberžė; Paberze; Paberzé
Perloja | foreign | Lithuania | Perloja
Eišiškės | foreign | Lithuania | Eišiškės; Eisiskes; Ejszyszki
Klaipeda | foreign | Lithuania | Klaipeda; Memel
Trakai | foreign | Lithuania | Trakai
Dotnuva | foreign | Lithuania | Dotnuva
# ── Poland ──
Warsaw | foreign | Poland | Warsaw; Warschau
Krakow | foreign | Poland | Krakow; Kraków; Krakau; Cracow
Gdansk | foreign | Poland | Gdansk; Gdańsk; Danzig
Lublin | foreign | Poland | Lublin
Lodz | foreign | Poland | Lodz; Łódź; L6dz
Poznan | foreign | Poland | Poznan; Poznań
Przemysl | foreign | Poland | Przemysl; Przemyśl
Wroclaw | foreign | Poland | Wroclaw; Wrocław; Breslau
Silesia | foreign | Poland | Silesia; Schlesien
Mazovia | foreign | Poland | Mazovia
Gdynia | foreign | Poland | Gdynia
Częstochowa | foreign | Poland | Czestochowa; Częstochowa
Sanok | foreign | Poland | Sanok
Pomerania | foreign | Poland | Pomerania; Pommern
# ── Moldova and Romania ──
Chișinău | foreign | Moldova | Kishinev; Chișinău; Chisinau; Chiinau
Transnistria | foreign | Moldova | Transnistria; Transdniestria; Transnistrien; Tiraspol
Bessarabia | foreign | Moldova | Bessarabia; Bessarabien
Bucharest | foreign | Romania | Bucharest; Bukarest
Cluj | foreign | Romania | Cluj; Cluj-Napoca
Timișoara | foreign | Romania | Timișoara; Timisoara
Transylvania | foreign | Romania | Transylvania; Siebenbürgen
Iași | foreign | Romania | Iasi; Iași; Jassy
# ── Central Europe and the Balkans ──
Berlin | foreign | Germany | Berlin
Hamburg | foreign | Germany | Hamburg
Munich | foreign | Germany | Munich; München
Frankfurt | foreign | Germany | Frankfurt
Hannover | foreign | Germany | Hannover
Potsdam | foreign | Germany | Potsdam
Dresden | foreign | Germany | Dresden
Cologne | foreign | Germany | Köln; Cologne
Leipzig | foreign | Germany | Leipzig
Düsseldorf | foreign | Germany | Düsseldorf
Nuremberg | foreign | Germany | Nürnberg; Nuremberg
Unna | foreign | Germany | Unna
Stuttgart | foreign | Germany | Stuttgart
Bavaria | foreign | Germany | Bavaria; Bayern
Prussia | foreign | historical | Prussia; East Prussia; Preußen; Ostpreußen
Vienna | foreign | Austria | Vienna; Wien
Prague | foreign | Czechia | Prague; Prag
Budapest | foreign | Hungary | Budapest
Szeged | foreign | Hungary | Szeged
Belgrade | foreign | Serbia | Belgrade; Belgrad
Novi Sad | foreign | Serbia | Novi Sad
Vojvodina | foreign | Serbia | Vojvodina
Zagreb | foreign | Croatia | Zagreb
Rijeka | foreign | Croatia | Rijeka; Fiume
Ljubljana | foreign | Slovenia | Ljubljana
Sarajevo | foreign | Bosnia | Sarajevo
Sofia | foreign | Bulgaria | Sofia
Plovdiv | foreign | Bulgaria | Plovdiv
Ruse | foreign | Bulgaria | Ruse
Balkans | foreign | historical | Balkans; Balkan
# ── Further afield ──
Istanbul | foreign | Turkey | Istanbul; Constantinople
Paris | foreign | France | Paris
London | foreign | United Kingdom | London
Rome | foreign | Italy | Rome; Rom
Venice | foreign | Italy | Venice; Venedig
Amsterdam | foreign | Netherlands | Amsterdam
Brussels | foreign | Belgium | Brussels; Brüssel
Washington | foreign | United States | Washington
Brooklyn | foreign | United States | Brooklyn; Manhattan; Brighton Beach
Boston | foreign | United States | Boston
Portland | foreign | United States | Portland
Los Angeles | foreign | United States | Los Angeles
Ulaanbaatar | foreign | Mongolia | Ulaanbaatar; Ulan Bator
Beijing | foreign | China | Beijing; Peking
Tehran | foreign | Iran | Tehran
Tbilisi | foreign | Georgia | Tbilisi; Tiflis
Narva | foreign | Estonia | Narva
Stockholm | foreign | Sweden | Stockholm
Jerusalem | foreign | Israel | Jerusalem
Tel Aviv | foreign | Israel | Tel Aviv
Toronto | foreign | Canada | Toronto
US states | foreign | United States | Texas; Florida; California; Pennsylvania; Minnesota; Montana; Oregon; Colorado; Ohio; Virginia; Alaska
US cities | foreign | United States | Philadelphia; Miami; Detroit; New Orleans; San Francisco; Newark; Las Vegas
Zurich | foreign | Switzerland | Zürich; Zurich
Madrid | foreign | Spain | Madrid
# ── States, empires, continents ──
Ukraine | state | Ukraine | Ukraine
Russia | state | Russia | Russia; Russland; Russian Federation
Soviet Union | state | historical | Soviet Union; USSR; Sowjetunion; UdSSR
Poland | state | Poland | Poland; Polen
Lithuania | state | Lithuania | Lithuania; Litauen
Belarus | state | Belarus | Belarus; Belorussia; Byelorussia; Weißrussland
Moldova | state | Moldova | Moldova; Moldau; Moldawien
Romania | state | Romania | Romania; Rumänien
Bulgaria | state | Bulgaria | Bulgaria; Bulgarien
Hungary | state | Hungary | Hungary; Ungarn
Serbia | state | Serbia | Serbia; Serbien
Croatia | state | Croatia | Croatia; Kroatien
Slovenia | state | Slovenia | Slovenia; Slowenien
Yugoslavia | state | historical | Yugoslavia; Jugoslawien
Czechoslovakia | state | historical | Czechoslovakia; Tschechoslowakei; Czech Republic; Tschechien
Austria | state | Austria | Austria; Österreich; Austro-Hungary; Austria-Hungary
Germany | state | Germany | Germany; Deutschland; West Germany
France | state | France | France; Frankreich
Italy | state | Italy | Italy; Italien
United Kingdom | state | United Kingdom | Britain; England; Großbritannien; United Kingdom
United States | state | United States | America; United States; USA; Amerika; Vereinigten Staaten
Turkey | state | Turkey | Turkey; Türkei
Greece | state | Greece | Greece; Griechenland
Latvia | state | Latvia | Latvia; Lettland
Estonia | state | Estonia | Estonia; Estland
Georgia | state | Georgia | Georgia; Georgien
Kazakhstan | state | Kazakhstan | Kazakhstan; Kasachstan
Mongolia | state | Mongolia | Mongolia; Mongolei
China | state | China | China
Israel | state | Israel | Israel
Syria | state | Syria | Syria; Syrien
Afghanistan | state | Afghanistan | Afghanistan
Sweden | state | Sweden | Sweden; Schweden
Canada | state | Canada | Canada; Kanada
Armenia | state | Armenia | Armenia; Armenien
Belgium | state | Belgium | Belgium; Belgien
Mexico | state | Mexico | Mexico; Mexiko
Japan | state | Japan | Japan
Iraq | state | Iraq | Iraq; Irak
Iran | state | Iran | Iran
Switzerland | state | Switzerland | Switzerland; Schweiz
Spain | state | Spain | Spain; Spanien
Finland | state | Finland | Finland; Finnland
Norway | state | Norway | Norway; Norwegen
Bosnia | state | Bosnia | Bosnia; Bosnien
Macedonia | state | Macedonia | Macedonia; Mazedonien
Slovakia | state | Slovakia | Slovakia; Slowakei
Uzbekistan | state | Uzbekistan | Uzbekistan; Usbekistan
Europe | state | continent | Europe; Europa
"""


def entries() -> list[dict]:
    out = []
    for line in RAW.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, kind, area, spellings = [p.strip() for p in line.split("|")]
        out.append(dict(name=name, kind=kind, area=area,
                        spellings=[s.strip() for s in spellings.split(";") if s.strip()]))
    return out


def compile_matcher():
    """One alternation, longest spelling first, so 'Saint Petersburg' beats 'Petersburg'."""
    lookup, spellings = {}, []
    for e in entries():
        for s in e["spellings"]:
            if s in lookup and lookup[s]["name"] != e["name"]:
                raise ValueError(f"spelling {s!r} listed under two places")
            lookup[s] = e
            spellings.append(s)
    spellings.sort(key=len, reverse=True)
    pattern = re.compile(
        r"(?<![\w\-])(" + "|".join(re.escape(s) for s in spellings) + r")(?:[’']s|s)?(?![\w])")
    return pattern, lookup


if __name__ == "__main__":
    es = entries()
    compile_matcher()
    from collections import Counter
    print(Counter(e["kind"] for e in es), len(es), "places,",
          sum(len(e["spellings"]) for e in es), "spellings")
    print(Counter(e["area"] for e in es if e["kind"] == "ua"))
