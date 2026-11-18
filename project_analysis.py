"""Анализ спортивного магазина.

Запуск: python project_analysis.py --data data.zip --personal personal_data.csv --out results
Зависимости: Python 3.12, pandas, numpy, scipy, scikit-learn, matplotlib.
Все денежные показатели выражены в единицах cost. Валюта в источнике не указана.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import unicodedata
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score, average_precision_score, brier_score_loss,
    calinski_harabasz_score, confusion_matrix, davies_bouldin_score,
    f1_score, log_loss, precision_score, recall_score, roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

SEED = 42
RNG = np.random.default_rng(SEED)


def log(s):
    print(s, flush=True)


def normalize(s):
    s = unicodedata.normalize("NFKC", str(s)).casefold().replace("ё", "е")
    s = s.replace("’", "'").replace("`", "'")
    return re.sub(r"\s+", " ", s).strip()


def parse_ids(s):
    # Переносы разрывают сами числа. Пробелы и прочие разделители сохраняются.
    values = list(map(int, re.findall(r"\d+", s.replace("\n", "").replace("\r", ""))))
    assert len(values) == len(set(values)) == 5000
    return set(values)


def colour_tokens(s):
    if pd.isna(s) or not str(s).strip():
        return ("не указан",)
    parts = [normalize(x) for x in str(s).split("/") if x.strip()]
    return tuple(sorted(set(parts))) or ("не указан",)


def product_category(s):
    if re.search(r"кроссов|кед[ыа ]|ботин|полубот|бутс|сандал|сабо|тапоч|шлепан|боксерк|чешк|борцовк|обув", s):
        return "обувь"
    if re.search(r"брюк|шорт|футбол|майк|куртк|ветровк|толстов|легинс|плать|юбк|купальник|плавк|лиф |джемпер|джерси|поло |жилет|трико|комбинез|носк|белье|термо|анорак|свит|лонгслив|топ |рашгард|костюм|перчатк|бейсбол|шапк|кепк", s):
        return "одежда"
    if re.search(r"велосипед|тренажер|дорожка|самокат|лыж|сноуборд|палатк|байдар|лодк|батут|коньк|ролик|ракетк|гантел|штанг|мяч|гиря|стол |стул|набор .*стол|мешок спальн|мешок боксер|бассейн|скамья|сапборд|гриф ", s):
        return "инвентарь"
    return "аксессуары и прочее"


def json_default(x):
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.floating): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (pd.Index, pd.Series)): return x.tolist()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)


def save_json(obj, path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def welch_result(test, control):
    t = np.asarray(test, float); c = np.asarray(control, float)
    vt = t.var(ddof=1)/len(t); vc = c.var(ddof=1)/len(c)
    se = np.sqrt(vt+vc)
    df = (vt+vc)**2/(vt*vt/(len(t)-1)+vc*vc/(len(c)-1)) if se else np.inf
    delta = t.mean()-c.mean()
    half = stats.t.ppf(.975, df)*se
    return dict(test_mean=t.mean(), control_mean=c.mean(), difference=delta,
                relative=delta/c.mean() if c.mean() else None,
                low=delta-half, high=delta+half,
                p=stats.ttest_ind(t,c,equal_var=False).pvalue)


def bootstrap_diff(test, control, n=3000):
    t=np.asarray(test,float);c=np.asarray(control,float);rng=np.random.default_rng(SEED)
    diffs=[];rels=[]
    for start in range(0,n,100):
        b=min(100,n-start)
        mt=t[rng.integers(len(t),size=(b,len(t)))].mean(axis=1)
        mc=c[rng.integers(len(c),size=(b,len(c)))].mean(axis=1)
        diffs.extend(mt-mc);rels.extend(mt/mc-1)
    return {"low":np.quantile(diffs,.025),"high":np.quantile(diffs,.975),
            "relative_low":np.quantile(rels,.025),"relative_high":np.quantile(rels,.975)}


def classification_metrics(y, proba, threshold=.5):
    pred=np.asarray(proba)>=threshold
    return dict(n=len(y),positive_rate=np.mean(y),f1=f1_score(y,pred),
                f1_macro=f1_score(y,pred,average="macro"),
                precision=precision_score(y,pred,zero_division=0),recall=recall_score(y,pred),
                roc_auc=roc_auc_score(y,proba),pr_auc=average_precision_score(y,proba),
                brier=brier_score_loss(y,proba),logloss=log_loss(y,proba),
                confusion=confusion_matrix(y,pred).tolist())


def load_data(data_zip, personal_path, out):
    out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(data_zip) as z:
        z.extractall(out/"input")
        pos=parse_ids(z.read("result_data/ids_first_company_positive.txt").decode())
        neg=parse_ids(z.read("result_data/ids_first_company_negative.txt").decode())
        compressed=pd.read_csv(z.open("result_data/personal_data.csv.gz"),compression="gzip")
    missing=pd.read_csv(personal_path) if personal_path else compressed
    pd.testing.assert_frame_equal(missing,compressed)
    with sqlite3.connect(out/"input/result_data/shop_database.db") as con:
        known=pd.read_sql_query("SELECT id, gender, age, education, city, country FROM personal_data",con)
        coeff=pd.read_sql_query("SELECT id, personal_coef FROM personal_data_coeffs",con)
        purchases=pd.read_sql_query("SELECT id, product, colour, cost, product_sex, base_sale, dt FROM purchases",con)
    assert not pos&neg
    people=pd.concat([known,missing],ignore_index=True)
    assert people.id.is_unique and coeff.id.is_unique
    assert set(people.id)==set(purchases.id)==set(coeff.id)
    n_all=len(people);n_p_all=len(purchases)
    people=people.query("country == 32").copy()
    people=people.merge(coeff,on="id",validate="one_to_one").set_index("id").sort_index()
    people["gender_observed"]=people.gender.notna().astype("int8")
    purchases=purchases[purchases.id.isin(people.index)].copy()
    # Единица наблюдения сохраняется: строка соответствует приобретению товара.
    # Нет order_id и quantity, поэтому число строк не называется числом заказов.
    purchases["product_raw"]=purchases["product"]
    purchases["product"]=purchases["product"].map({s:normalize(s) for s in purchases["product"].unique()})
    tokens={s:colour_tokens(s) for s in purchases.colour.dropna().unique()}
    purchases["colour_components"]=purchases.colour.map(tokens).apply(lambda x:x if isinstance(x,tuple) else ("не указан",))
    purchases["colour"]=purchases.colour_components.map(lambda x:"/".join(x))
    purchases["colour_count"]=purchases.colour_components.map(len)
    purchases["colour_missing"]=(purchases.colour=="не указан").astype("int8")
    purchases["product_sex_missing"]=purchases.product_sex.isna().astype("int8")
    # Отсутствие значения не приравнивается к доказанной универсальности товара.
    purchases["product_sex"]=purchases.product_sex.fillna(-1).astype("int8")
    purchases["cost_raw"]=purchases.cost
    purchases["cost_corrected"]=(purchases.cost<0).astype("int8")
    purchases["cost"]=purchases.cost.clip(lower=0)
    purchases["category"]=purchases["product"].map({s:product_category(s) for s in purchases["product"].unique()})
    assert purchases.isna().sum().sum()==0
    assert len(purchases)==purchases.merge(people.reset_index(),on="id",validate="many_to_one").shape[0]
    quality=dict(all_clients=n_all,clients=len(people),excluded_clients=n_all-len(people),
        observed_gender=int(people.gender.notna().sum()),missing_gender=int(people.gender.isna().sum()),
        all_purchases=n_p_all,purchases=len(purchases),days=[int(purchases.dt.min()),int(purchases.dt.max())],
        raw_products=purchases.product_raw.nunique(),clean_products=purchases["product"].nunique(),
        missing_colour=int(purchases.colour_missing.sum()),missing_product_sex=int(purchases.product_sex_missing.sum()),
        negative_cost=int(purchases.cost_corrected.sum()),cost_correction=int((purchases.cost-purchases.cost_raw).sum()),
        revenue=int(purchases.cost.sum()),raw_revenue=int(purchases.cost_raw.sum()),
        transaction_duplicates=int(purchases[["id","product_raw","colour","cost_raw","dt","base_sale"]].duplicated().sum()),
        test_all=len(pos),control_all=len(neg),test_country32=len(pos&set(people.index)),control_country32=len(neg&set(people.index)),
        city1188=int((people.city==1188).sum()),city1134=int((people.city==1134).sum()),csv_equal=True)
    log(f"Подготовка: {quality}")
    save_json(quality,out/"quality.json")
    people.to_pickle(out/"people_initial.pkl");purchases.to_pickle(out/"purchases.pkl")
    save_json({"test":sorted(pos),"control":sorted(neg)},out/"assignments.json")
    return people,purchases,pos,neg,quality


def aggregate_history(p, ids, end_day):
    p=p[p.dt<end_day]
    g=p.groupby("id",observed=True)
    d=g.agg(items=("cost","size"),spend=("cost","sum"),mean_price=("cost","mean"),
            last_day=("dt","max"),active_days=("dt","nunique"),sale_share=("base_sale","mean"))
    d["recency"]=end_day-d.pop("last_day")
    d=d.reindex(ids)
    d["history_empty"]=d["items"].isna().astype("int8")
    d["recency"]=d.recency.fillna(end_day+1)
    d=d.fillna(0)
    for sex,label in [(0,"female_share"),(1,"male_share"),(-1,"unknown_sex_share")]:
        n=p[p.product_sex==sex].groupby("id").size().reindex(ids,fill_value=0)
        d[label]=(n/d["items"].replace(0,1)).fillna(0)
    for cat,label in [("одежда","clothing_share"),("обувь","footwear_share"),("инвентарь","equipment_share")]:
        n=p[p.category==cat].groupby("id").size().reindex(ids,fill_value=0)
        d[label]=(n/d["items"].replace(0,1)).fillna(0)
    return d


def gender_model(people,p,out):
    h=aggregate_history(p,people.index,int(p.dt.max())+1)
    x=people[["age","personal_coef"]].copy()
    x["education_high"]=(people.education=="высшее").astype(int)
    # Только personal_coef, как указано в задании. Другие закрытые коэффициенты не используются.
    x=x.join(h.drop(columns="history_empty"))
    x["log_items"]=np.log1p(x.pop("items"));x["log_spend"]=np.log1p(x.pop("spend"))
    x["log_price"]=np.log1p(x.pop("mean_price"))
    labelled=people[people.gender.notna()].index
    tr,rest=train_test_split(labelled,test_size=.4,stratify=people.loc[labelled,"gender"],random_state=SEED)
    va,te=train_test_split(rest,test_size=.5,stratify=people.loc[rest,"gender"],random_state=SEED)
    y=people.gender.astype("Int64")
    models={
      "Логистическая регрессия":make_pipeline(StandardScaler(),LogisticRegression(C=10,max_iter=2000,random_state=SEED)),
      "Случайный лес":RandomForestClassifier(n_estimators=200,max_depth=16,min_samples_leaf=3,max_features=.8,n_jobs=4,random_state=SEED),
      "Градиентный бустинг":HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,l2_regularization=2,learning_rate=.08,random_state=SEED),
    }
    results={};best=None;bestscore=-1
    for name,model in models.items():
        model.fit(x.loc[tr],y.loc[tr].astype(int));proba=model.predict_proba(x.loc[va])[:,1]
        thresholds=np.linspace(.2,.8,61)
        fs=np.array([f1_score(y.loc[va].astype(int),proba>=t,average="macro") for t in thresholds])
        threshold=float(thresholds[np.flatnonzero(fs==fs.max())[np.argmin(abs(thresholds[fs==fs.max()]-.5))]])
        met=classification_metrics(y.loc[va].astype(int),proba,threshold)
        results[name]={"validation":met,"threshold":threshold}
        log(f"Пол: {name}, validation macro F1={met['f1_macro']:.6f}, threshold={threshold}")
        if met["f1_macro"]>bestscore:best=(name,model,threshold);bestscore=met["f1_macro"]
    name,model,threshold=best
    # Тест открывается один раз после выбора модели и порога по validation.
    trainval=tr.append(va);model.fit(x.loc[trainval],y.loc[trainval].astype(int))
    met=classification_metrics(y.loc[te].astype(int),model.predict_proba(x.loc[te])[:,1],threshold)
    log(f"Пол: итоговый независимый тест {met}")
    blind=people[people.gender.isna()].index
    model.fit(x.loc[labelled],y.loc[labelled].astype(int))
    probs=model.predict_proba(x.loc[blind])[:,1]
    people.loc[blind,"gender"]=(probs>=threshold).astype(int)
    people["gender"]=people.gender.astype("int8")
    people["gender_probability"]=1.0
    people.loc[blind,"gender_probability"]=np.maximum(probs,1-probs)
    result={"selected":name,"threshold":threshold,"train":len(tr),"validation":len(va),"test":met,
      "comparisons":results,"restored":len(blind),"restored_male":int((probs>=threshold).sum()),
      "restored_low_confidence":int((np.maximum(probs,1-probs)<.8).sum()),"features":list(x.columns)}
    # Вероятность у наблюдённых значений является флагом известности, а не оценкой модели.
    save_json(result,out/"gender.json")
    people.to_pickle(out/"people_gender.pkl")
    return people,h,result


def experiment(people,p,pos,neg,out):
    cohort=people.loc[people.index.isin(pos|neg)].copy()
    cohort["group"]=np.where(cohort.index.isin(pos),"test","control")
    def metric_frame(start,end):
        q=p[p.dt.between(start,end)]
        d=q.groupby("id").agg(revenue=("cost","sum"),items=("cost","size"),
                              active_days=("dt","nunique"),discount_items=("base_sale","sum"))
        d=d.reindex(cohort.index,fill_value=0)
        d["buyer"]=(d["items"]>0).astype(int)
        return d.join(cohort[["group","age","personal_coef","education","city"]])
    main=metric_frame(5,16);pre=metric_frame(0,4)
    results={}
    for col in ["revenue","buyer","items","active_days"]:
        results[col]=welch_result(main.loc[main.group=="test",col],main.loc[main.group=="control",col])
    results["revenue"]["bootstrap"]=bootstrap_diff(main.loc[main.group=="test","revenue"],main.loc[main.group=="control","revenue"])
    # Holm для трёх дополнительных проверок, основная метрика задана заранее.
    sec=["buyer","items","active_days"];order=sorted(sec,key=lambda k:results[k]["p"]);prev=0
    for i,k in enumerate(order):
        prev=max(prev,min(1,results[k]["p"]*(len(sec)-i)));results[k]["p_holm"]=prev
    summaries={}
    for group,d in main.groupby("group"):
        summaries[group]=dict(n=len(d),buyers=int(d.buyer.sum()),conversion=d.buyer.mean(),
          revenue=d.revenue.sum(),revenue_per_user=d.revenue.mean(),items=int(d["items"].sum()),
          items_per_user=d["items"].mean(),revenue_per_buyer=d.revenue.sum()/d.buyer.sum(),
          mean_item_price=d.revenue.sum()/d["items"].sum(),sale_share=d.discount_items.sum()/d["items"].sum())
    baseline={col:welch_result(pre.loc[pre.group=="test",col],pre.loc[pre.group=="control",col]) for col in ["revenue","buyer","items","age","personal_coef"]}
    # Приведение к выручке на клиента за один день устраняет разную длину окон.
    change=main.revenue/12-pre.revenue/5
    did=welch_result(change[cohort.group=="test"],change[cohort.group=="control"])
    did["bootstrap"]=bootstrap_diff(change[cohort.group=="test"],change[cohort.group=="control"])
    sensitivity={}
    for label,start,end,exclude1134 in [("без города 1134",5,16,True),("сдвиг на один день",4,15,False),("без последнего дня",5,15,False)]:
        d=metric_frame(start,end)
        if exclude1134:d=d[d.city!=1134]
        sensitivity[label]={"n_test":int((d.group=="test").sum()),"n_control":int((d.group=="control").sum()),
                            **welch_result(d.loc[d.group=="test","revenue"],d.loc[d.group=="control","revenue"])}
    balance=[]
    for col in ["age","personal_coef"]:
        t=cohort.loc[cohort.group=="test",col];c=cohort.loc[cohort.group=="control",col]
        balance.append(dict(feature=col,test=t.mean(),control=c.mean(),smd=(t.mean()-c.mean())/np.sqrt((t.var()+c.var())/2)))
    for label,series in [("высшее образование",cohort.education=="высшее"),("город 1134",cohort.city==1134),("город 1188",cohort.city==1188)]:
        t=series[cohort.group=="test"].astype(float);c=series[cohort.group=="control"].astype(float)
        balance.append(dict(feature=label,test=t.mean(),control=c.mean(),smd=(t.mean()-c.mean())/np.sqrt((t.var()+c.var())/2)))
    tp=pre.loc[pre.group=="test","revenue"];cp=pre.loc[pre.group=="control","revenue"]
    balance.append(dict(feature="выручка до кампании",test=tp.mean(),control=cp.mean(),smd=(tp.mean()-cp.mean())/np.sqrt((tp.var()+cp.var())/2)))
    daily=p[p.id.isin(cohort.index)].groupby(["dt","id"]).agg(revenue=("cost","sum"),items=("cost","size")).reset_index()
    daily["group"]=np.where(daily.id.isin(pos),"test","control")
    daily=daily.groupby(["dt","group"]).agg(revenue=("revenue","sum"),items=("items","sum"),buyers=("id","size")).reset_index()
    daily["n"]=daily.group.map(cohort.group.value_counts())
    daily["revenue_per_user"]=daily.revenue/daily.n
    daily.to_csv(out/"experiment_daily.csv",index=False)
    main.reset_index().to_pickle(out/"experiment_users.pkl")
    result={"window":[5,16],"baseline_window":[0,4],"alpha":.05,"summaries":summaries,"metrics":results,
      "baseline":baseline,"balance":balance,"difference_in_differences_daily":did,"sensitivity":sensitivity,
      "sample_ratio_p":stats.chisquare([summaries["test"]["n"],summaries["control"]["n"]]).pvalue,
      "causal_limit":"Нет подтверждения случайного назначения. Группы различаются до кампании. Оценки являются сравнением групп.",
      "units":"денежные единицы cost"}
    save_json(result,out/"experiment.json")
    log(f"Кампания: {json.dumps(result,ensure_ascii=False,default=json_default)}")
    city=p.merge(people[["city"]],left_on="id",right_index=True,validate="many_to_one")
    city["audience"]=np.where(city.city==1134,"1134","прочие")
    citydaily=city.groupby(["dt","audience"]).agg(revenue=("cost","sum"),items=("cost","size"),buyers=("id","nunique")).reset_index()
    sizes={"1134":int((people.city==1134).sum()),"прочие":int((people.city!=1134).sum())}
    citydaily["n"]=citydaily.audience.map(sizes);citydaily["revenue_per_user"]=citydaily.revenue/citydaily.n
    citydaily.to_csv(out/"billboard_daily.csv",index=False)
    return result


def clusters(people,h,p,out,chosen=None):
    x=h[["recency","items","mean_price","sale_share","clothing_share","footwear_share","equipment_share"]].copy()
    x["items"]=np.log1p(x["items"]);x["mean_price"]=np.log1p(x["mean_price"])
    # Географические коды и предсказанный пол не задают расстояния между клиентами.
    scaler=StandardScaler();z=scaler.fit_transform(x)
    idx=RNG.choice(len(z),size=min(40000,len(z)),replace=False);sample=z[idx]
    eval_idx=RNG.choice(len(sample),size=min(4000,len(sample)),replace=False)
    scores=[];models={}
    for k in range(2,8):
        m=KMeans(n_clusters=k,n_init=10,max_iter=200,random_state=SEED).fit(sample)
        lab=m.predict(sample)
        sil=silhouette_score(sample[eval_idx],lab[eval_idx])
        other=KMeans(n_clusters=k,n_init=5,max_iter=200,random_state=73).fit(sample)
        stability=adjusted_rand_score(lab,other.predict(sample))
        scores.append(dict(k=k,silhouette=sil,davies_bouldin=davies_bouldin_score(sample,lab),
          calinski_harabasz=calinski_harabasz_score(sample,lab),inertia=m.inertia_,
          stability_ari=stability,smallest_share=np.bincount(lab).min()/len(lab)))
        models[k]=m;log(f"Кластеры: {scores[-1]}")
    # Выбор на основе silhouette с ограничением минимальной доли сегмента.
    valid=[s for s in scores if s["smallest_share"]>=.03]
    k=chosen or max(valid,key=lambda s:s["silhouette"])["k"]
    model=models[k];labels=model.predict(z)
    # Стабильная нумерация по средней выручке на клиента.
    order=h.assign(label=labels).groupby("label").spend.mean().sort_values().index
    mapping={int(old):i+1 for i,old in enumerate(order)}
    people["cluster"]=pd.Series(labels,index=people.index).map(mapping).astype(int)
    allh=people.join(h)
    prof=allh.groupby("cluster").agg(clients=("age","size"),age=("age","mean"),
      male_share=("gender","mean"),mean_spend=("spend","mean"),total_spend=("spend","sum"),
      items=("items","mean"),mean_price=("mean_price","mean"),recency=("recency","mean"),
      sale_share=("sale_share","mean"),clothing_share=("clothing_share","mean"),
      footwear_share=("footwear_share","mean"),equipment_share=("equipment_share","mean"))
    prof["client_share"]=prof.clients/len(people);prof["revenue_share"]=prof.total_spend/prof.total_spend.sum()
    pp=p.merge(people[["cluster"]],left_on="id",right_index=True,validate="many_to_one")
    prof["discount_item_share"]=pp.groupby("cluster").base_sale.mean()
    prof["total_items"]=pp.groupby("cluster").size()
    top=pp.groupby(["cluster","product"]).agg(items=("id","size"),buyers=("id","nunique"),revenue=("cost","sum")).reset_index()
    top=top.sort_values(["cluster","buyers"],ascending=[True,False]).groupby("cluster").head(5)
    top["penetration"]=top.buyers/top.cluster.map(prof.clients)
    category=pd.crosstab(pp.cluster,pp.category,normalize="index")
    sale=pp.groupby(["cluster","base_sale"]).agg(items=("id","size"),buyers=("id","nunique"),mean_price=("cost","mean"),revenue=("cost","sum")).reset_index()
    # Экспериментальный срез внутри сегментов описательный: кластеры построены после кампании.
    prof.to_csv(out/"cluster_profiles.csv");top.to_csv(out/"cluster_products.csv",index=False);category.to_csv(out/"cluster_categories.csv");sale.to_csv(out/"cluster_sales.csv",index=False)
    people.to_pickle(out/"people_cluster.pkl")
    result={"selected_k":k,"fit_clients":len(sample),"evaluation_clients":len(eval_idx),"features":list(x.columns),
            "metrics":scores,"profiles":prof.reset_index().to_dict("records"),"top_products":top.to_dict("records"),
            "discount_caveat":"Доля покупок со скидкой является описательной связью. Нет данных о показах предложений и доступности скидки для непокупателей."}
    save_json(result,out/"clusters.json")
    return people,result


def marketing_review(people,p,pos,neg,out):
    """Описательное сравнение скидок и двух дней рекламы на билбордах."""
    assert people.index.is_unique and set(people.country)=={32}
    pp=p.merge(people[["cluster","city"]],left_on="id",right_index=True,validate="many_to_one")
    assert len(pp)==len(p) and set(pp.base_sale.unique())=={0,1}
    sales=pp.groupby(["cluster","base_sale"]).agg(
        items=("id","size"),buyers=("id","nunique"),
        mean_price=("cost","mean"),revenue=("cost","sum")).reset_index()
    sizes=people.groupby("cluster").size()
    sales["items_per_buyer"]=sales["items"]/sales.buyers
    sales["segment_clients"]=sales.cluster.map(sizes)
    sales["items_per_segment_client"]=sales["items"]/sales.segment_clients
    comparisons=[]
    for cluster,d in sales.groupby("cluster"):
        d=d.set_index("base_sale");plain=d.loc[0];sale=d.loc[1]
        comparisons.append(dict(cluster=int(cluster),
            mean_price_change=sale.mean_price/plain.mean_price-1,
            items_per_buyer_change=sale.items_per_buyer/plain.items_per_buyer-1))
    assert sales["items"].sum()==len(p)
    assert np.isclose(sales.revenue.sum(),p.cost.sum())
    sales.to_csv(out/"cluster_sales.csv",index=False)

    def daily_for(excluded):
        profiles=people.loc[~people.index.isin(excluded)]
        q=pp.loc[pp.id.isin(profiles.index)].copy()
        q["audience"]=np.where(q.city==1134,"1134","прочие")
        daily=q.groupby(["dt","audience"]).agg(
            revenue=("cost","sum"),items=("id","size"),buyers=("id","nunique"))
        idx=pd.MultiIndex.from_product([range(int(p.dt.min()),int(p.dt.max())+1),["1134","прочие"]],names=["dt","audience"])
        daily=daily.reindex(idx,fill_value=0).reset_index()
        n={"1134":int((profiles.city==1134).sum()),"прочие":int((profiles.city!=1134).sum())}
        daily["n"]=daily.audience.map(n)
        daily["revenue_per_user"]=daily.revenue/daily.n
        return daily

    def compare_days(daily):
        records=[]
        for day in [15,45]:
            for audience in ["1134","прочие"]:
                before=daily[(daily.audience==audience)&daily.dt.between(day-7,day-1)]
                event=daily[(daily.audience==audience)&(daily.dt==day)].iloc[0]
                assert len(before)==7 and before.n.nunique()==1
                records.append(dict(day=day,audience=audience,n=int(event.n),
                    baseline_days=[day-7,day-1],revenue=float(event.revenue),
                    revenue_per_user=float(event.revenue_per_user),
                    baseline_revenue_per_user=float(before.revenue_per_user.mean()),
                    revenue_growth=float(event.revenue_per_user/before.revenue_per_user.mean()-1),
                    items=int(event["items"]),baseline_daily_items=float(before["items"].mean()),
                    items_growth=float(event["items"]/before["items"].mean()-1),
                    buyers=int(event.buyers),baseline_daily_buyers=float(before.buyers.mean()),
                    buyer_rate=float(event.buyers/event.n),
                    baseline_daily_buyer_rate=float(before.buyers.mean()/event.n)))
        return records

    daily=daily_for(set());daily.to_csv(out/"billboard_daily.csv",index=False)
    result=dict(window=[int(p.dt.min()),int(p.dt.max())],discount_comparison=sales.to_dict("records"),
        discount_changes=comparisons,billboard_comparison=compare_days(daily),
        billboard_without_email_participants=compare_days(daily_for(pos|neg)),
        price_definition="Средняя стоимость строки покупки. ID заказа отсутствует, средний чек заказа не вычисляется.",
        frequency_definition="Число строк данного типа, делённое на число покупателей этого типа. Один клиент может входить в обе группы.",
        discount_limit="Различаются товары, доступность скидки и состав покупателей. Доля скидок входит в кластеризацию. Причинный эффект не определяется.",
        billboard_limit="Дни 15 и 45 сравниваются со средним за семь предыдущих дней и другими городами страны 32. Нет случайного назначения. Реклама совпадает со скидкой, день 15 пересекается с рассылкой. Город означает место рождения.")
    save_json(result,out/"marketing_review.json")
    log("Дополнительные сравнения: 6 строк по скидкам, 2 дня кампании на билбордах, проверка без участников рассылки.")
    return result


def rank_metrics(y, proba, fraction=.1):
    y=np.asarray(y);proba=np.asarray(proba)
    n=max(1,int(np.ceil(len(y)*fraction)))
    top=np.argsort(-proba,kind="stable")[:n]
    return dict(n=len(y),positives=int(y.sum()),base_rate=y.mean(),roc_auc=roc_auc_score(y,proba),
      pr_auc=average_precision_score(y,proba),brier=brier_score_loss(y,proba),logloss=log_loss(y,proba),
      mean_probability=proba.mean(),precision_top10=y[top].mean(),
      recall_top10=y[top].sum()/max(1,y.sum()),lift_top10=y[top].mean()/y.mean())


def pair_features(people,p,catalog,start,pos,neg,score_email=False,only_ids=None):
    ids=people.index if only_ids is None else pd.Index(only_ids)
    h=aggregate_history(p,ids,start)
    h["age"]=people.loc[ids,"age"]
    h["personal_coef"]=people.loc[ids,"personal_coef"]
    h["education_high"]=(people.loc[ids,"education"]=="высшее").astype(int)
    h["city1134"]=(people.loc[ids,"city"]==1134).astype(int)
    h["city1188"]=(people.loc[ids,"city"]==1188).astype(int)
    for col in ["items","spend","mean_price","active_days"]:
        h["log_"+col]=np.log1p(h[col])
    h=h.drop(columns=["items","spend","mean_price","active_days"])
    pairs=pd.MultiIndex.from_product([ids,catalog.index],names=["id","product"])
    d=pd.DataFrame(index=pairs).reset_index()
    d=d.merge(h,left_on="id",right_index=True,validate="many_to_one")
    hist=p[(p.dt<start)&p["product"].isin(catalog.index)]
    hp=hist.groupby(["id","product"]).agg(prior_product_items=("dt","size"),prior_product_last=("dt","max"),prior_product_sale=("base_sale","mean"))
    d=d.join(hp,on=["id","product"],validate="one_to_one")
    d["prior_product_recency"]=start-d.pop("prior_product_last").fillna(-1)
    d["prior_product_items"]=d.prior_product_items.fillna(0)
    d["prior_product_sale"]=d.prior_product_sale.fillna(0)
    d["prior_product_items"]=np.log1p(d.prior_product_items)
    pop=hist.groupby("product").id.nunique()/len(people)
    d["product_popularity"]=d["product"].map(pop).fillna(0)
    d=d.merge(catalog,left_on="product",right_index=True,validate="many_to_one")
    d["price_affinity"]=d["log_price"]-d["log_mean_price"]
    d["sex_affinity"]=np.where(d.product_sex==1,d.male_share,np.where(d.product_sex==0,d.female_share,d.unknown_sex_share))
    d["category_affinity"]=np.select([d.category=="одежда",d.category=="обувь",d.category=="инвентарь"],
                                    [d.clothing_share,d.footwear_share,d.equipment_share],default=1-d.clothing_share-d.footwear_share-d.equipment_share)
    d["email_offer"]=1 if score_email else ((d.id.isin(pos))&(start==5)).astype(int)
    d["billboard_offer"]=(d.city1134.astype(bool)&any(start<=day<=start+11 for day in [15,45])).astype(int)
    for i,name in enumerate(catalog.index):d[f"product_{i}"]=(d["product"]==name).astype(int)
    for sex,label in [(-1,"unknown"),(0,"female"),(1,"male")]:d["item_sex_"+label]=(d.product_sex==sex).astype(int)
    # Таргет использует только покупки внутри будущего окна.
    outcome=p[p.dt.between(start,start+11)&p["product"].isin(catalog.index)].groupby(["id","product"]).size()
    d["target"]=pd.MultiIndex.from_frame(d[["id","product"]]).isin(outcome.index).astype("int8")
    meta=d[["id","product","target"]].copy()
    features=d.drop(columns=["id","product","target","product_sex","category"])
    assert features.isna().sum().sum()==0
    return features.astype("float32"),meta


def propensity(people,p,pos,neg,out):
    # Каталог фиксируется до первого прогнозного окна, без просмотра будущего спроса.
    early=p[p.dt<5]
    products=early.groupby("product").id.nunique().sort_values(ascending=False).head(8).index
    catalog=early[early["product"].isin(products)].groupby("product").agg(
        price=("cost","median"),product_sex=("product_sex",lambda s:s.mode().iloc[0]),
        category=("category",lambda s:s.mode().iloc[0]),missing_colour_rate=("colour_missing","mean"),
        colour_count_mean=("colour_count","mean"))
    catalog["log_price"]=np.log1p(catalog.pop("price"))
    catalog=catalog.reindex(products)
    trainids=np.random.default_rng(SEED).choice(people.index,size=min(40000,len(people)),replace=False)
    train_x=[];train_y=[]
    for start in [5,17]:
        x,m=pair_features(people,p,catalog,start,pos,neg,only_ids=trainids)
        train_x.append(x);train_y.append(m.target)
        log(f"Склонность: train окно {start}, строк {len(x)}, покупок {m.target.sum()}")
    tx=pd.concat(train_x,ignore_index=True);ty=pd.concat(train_y,ignore_index=True)
    vx,vm=pair_features(people,p,catalog,29,pos,neg,only_ids=trainids)
    models={
      "Логистическая регрессия":make_pipeline(StandardScaler(),LogisticRegression(C=.3,max_iter=400,random_state=SEED)),
      "Градиентный бустинг":HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,
          min_samples_leaf=100,l2_regularization=20,learning_rate=.06,early_stopping=False,random_state=SEED),
    }
    comparisons={};selected=None;best=-1
    for name,model in models.items():
        model.fit(tx,ty)
        pred=model.predict_proba(vx)[:,1]
        met=rank_metrics(vm.target,pred)
        comparisons[name]=met
        log(f"Склонность: validation {name}: {met}")
        if met["pr_auc"]>best:selected=(name,model);best=met["pr_auc"]
    name,model=selected
    # После выбора модели используется весь доступный прошлый validation период.
    fitx=pd.concat([tx,vx],ignore_index=True);fity=pd.concat([ty,vm.target],ignore_index=True)
    model.fit(fitx,fity)
    testx,testm=pair_features(people,p,catalog,45,pos,neg)
    testp=model.predict_proba(testx)[:,1]
    testm["probability"]=testp;testm["city"]=testm.id.map(people.city)
    testmetrics=rank_metrics(testm.target,testp)
    by_city={}
    for label,mask in [("1188",testm.city==1188),("1134",testm.city==1134),("прочие",~testm.city.isin([1188,1134]))]:
        by_city[label]=rank_metrics(testm.loc[mask,"target"],testp[mask])
    by_product=[]
    for pr,df in testm.groupby("product"):
        by_product.append({"product":pr,**rank_metrics(df.target,df.probability)})
    # Калибровка проверяется по десяти равным по числу строк диапазонам прогнозов.
    testm["decile"]=pd.qcut(testm.probability.rank(method="first"),10,labels=False)+1
    calibration=testm.groupby("decile").agg(probability=("probability","mean"),actual=("target","mean"),n=("target","size"))
    policy={}
    for label,q in [("1188",testm[testm.city==1188]),("all",testm)]:
        best=q.sort_values("probability",ascending=False).drop_duplicates("id")
        n=int(np.ceil(len(best)*.1));top=best.head(n)
        policy[label]=dict(clients=len(best),precision_all=best.target.mean(),top10_clients=n,
          top10_buyers=int(top.target.sum()),precision_top10=top.target.mean(),mean_score_top10=top.probability.mean())
    q=testm[(testm.city==1188)&(testm["product"]==products[0])].sort_values("probability",ascending=False)
    n=int(np.ceil(len(q)*.1))
    primary_product=dict(product=products[0],n=len(q),positives=int(q.target.sum()),base_rate=q.target.mean(),
      top10_n=n,top10_positives=int(q.head(n).target.sum()),precision_top10=q.head(n).target.mean(),
      lift_top10=q.head(n).target.mean()/q.target.mean())
    testm.to_pickle(out/"propensity_test.pkl")
    calibration.to_csv(out/"propensity_calibration.csv")
    log(f"Склонность: независимый будущий тест {testmetrics}, город1188 {by_city['1188']}")
    # Модель не переобучается по тестовой метрике. Тестовые наблюдения добавляются
    # только в финальную обучающую базу после фиксации результатов оценки.
    x45,m45=pair_features(people,p,catalog,45,pos,neg,only_ids=trainids)
    allx=pd.concat([fitx,x45],ignore_index=True);ally=pd.concat([fity,m45.target],ignore_index=True)
    model.fit(allx,ally)
    target_ids=people.index[people.city==1188]
    sx,sm=pair_features(people,p,catalog,59,pos,neg,score_email=True,only_ids=target_ids)
    sp=model.predict_proba(sx)[:,1]
    sm["purchase_probability"]=sp;sm=sm.drop(columns="target")
    sm["rank_for_client"]=sm.groupby("id").purchase_probability.rank(method="first",ascending=False).astype(int)
    sm["rank_for_product"]=sm.groupby("product").purchase_probability.rank(method="first",ascending=False).astype(int)
    sm.sort_values(["id","rank_for_client"]).to_csv(out/"propensity_scores.csv.gz",index=False,compression="gzip")
    chosen=sm[sm.rank_for_client==1].sort_values("purchase_probability",ascending=False).copy()
    chosen["priority_rank"]=np.arange(1,len(chosen)+1)
    chosen.to_csv(out/"recommendations.csv.gz",index=False,compression="gzip")
    recommendations=chosen.groupby("product").agg(clients=("id","size"),mean_probability=("purchase_probability","mean")).sort_values("clients",ascending=False)
    top10=chosen.head(int(np.ceil(len(chosen)*.1)))
    result=dict(selected=name,train_windows=[[5,16],[17,28]],validation_window=[29,40],test_window=[45,56],
      scoring_window=[59,70],train_clients=len(trainids),train_rows=len(tx),validation_rows=len(vx),test_rows=len(testx),
      catalog=list(products),comparisons=comparisons,test=testmetrics,test_city=by_city,test_product=by_product,
      features=list(tx.columns),target_clients=len(target_ids),scored_pairs=len(sm),
      top10_clients=len(top10),top10_mean_score=top10.purchase_probability.mean(),
      recommendation_products=recommendations.reset_index().to_dict("records"),
      one_offer_policy=policy,city1188_primary_product=primary_product,
      target_definition="Хотя бы одно приобретение конкретного товара в следующие 12 дней",
      communication_definition="Назначение персонального предложения или принадлежность к аудитории билбордов",
      limits="Вероятность покупки не равна дополнительному эффекту коммуникации. Нет логов доставки и просмотра, размера персональной скидки, себестоимости. Новая кампания требует контрольной группы.")
    save_json(result,out/"propensity.json")
    log(f"Прогноз для города1188: {recommendations.to_dict('index')}")


def export_final(people,p,out):
    if (out/"people_cluster.pkl").exists():people=pd.read_pickle(out/"people_cluster.pkl")
    # Итоговая витрина сохраняет исходную детализацию покупок и очищенный текст.
    profile=people.drop(columns="gender_probability")
    final=p.merge(profile,left_on="id",right_index=True,validate="many_to_one")
    final["colour_components"]=final.colour_components.map(lambda x:"|".join(x))
    assert final.isna().sum().sum()==0 and final.id.nunique()==len(people)
    assert len(final)==len(p) and not final.duplicated().any()
    final.to_csv(out/"final_dataframe.csv.gz",compression="gzip",index=False)
    log(f"Итоговый датафрейм: {final.shape}, пропусков {final.isna().sum().sum()}")


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",default="upload/data.zip");ap.add_argument("--personal",default="upload/personal_data.csv")
    ap.add_argument("--out",default="analysis/results");ap.add_argument("--stage",default="all");args=ap.parse_args();out=Path(args.out)
    with threadpool_limits(limits=4):
        if args.stage in ["all","prepare"]:
            people,p,pos,neg,quality=load_data(args.data,args.personal,out)
        else:
            p=pd.read_pickle(out/"purchases.pkl");people=pd.read_pickle(out/"people_initial.pkl")
            ass=json.loads((out/"assignments.json").read_text());pos=set(ass["test"]);neg=set(ass["control"])
        if args.stage=="review":
            marketing_review(pd.read_pickle(out/"people_cluster.pkl"),p,pos,neg,out)
            return
        if args.stage in ["all","gender"]:people,h,g=gender_model(people,p,out)
        else:
            if (out/"people_gender.pkl").exists():people=pd.read_pickle(out/"people_gender.pkl")
            h=aggregate_history(p,people.index,int(p.dt.max())+1)
        if args.stage in ["all","experiment"]:experiment(people,p,pos,neg,out)
        if args.stage in ["all","clusters"]:people,c=clusters(people,h,p,out)
        if args.stage in ["all","propensity"]:
            propensity(people,p,pos,neg,out)
        if args.stage in ["all","export"]:export_final(people,p,out)
        if args.stage=="all":marketing_review(people,p,pos,neg,out)


if __name__=="__main__":
    main()
