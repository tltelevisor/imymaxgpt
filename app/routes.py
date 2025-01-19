from flask import render_template, flash, redirect, url_for, request, jsonify
from app.forms import LoginForm, RegistrationForm, PostForm, ProductsForm, NewFAQ, EditFAQ
from flask_login import current_user, login_user, logout_user, login_required
from app.models import (User, Post, Products, Files, rolepr, Faq, Batch, Catgr, Answ_faq,
                        catgr_files, catgr_batches, prd_cat_faq, Topic)
from datetime import datetime
from sqlalchemy import insert, delete
from os import remove, makedirs, path
from app import brand, brand_gpt
from openai import OpenAI
import openai, tiktoken
from app import app, db, gl_api_key
from scipy import spatial
from pgpt_python.client import PrivateGPTApi
import pandas as pd
import pickle, ast
from sqlalchemy import select, and_
from numpy import round
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from pydantic import BaseModel


def check_openai_api_key(api):
    try:
        client = OpenAI(api_key=api)
        client.models.list()
    except openai.AuthenticationError as err:
        app.logger.error(f'err: {err}, api_key: {gl_api_key}')
        return False
    except openai.PermissionDeniedError as err:
        app.logger.error(f'err: {err}')
        return False
    else:
        return True


def set_openai_api_key(user, u_api_key):
    global gl_api_key
    gl_api_key = u_api_key
    app.logger.info(f'Установлен введенный пользователем {user} API_KEY: {u_api_key}')
    return


def serv_status():
    if app.config['LLM'] == 'OpenAI':
        if check_openai_api_key(gl_api_key):
            status = [0, f'(oai)']
        else:
            status = [1, f'(oai)']
    elif app.config['LLM'] == 'PrivateGPT':
        try:
            client = PrivateGPTApi(base_url=app.config['URL_PGPT'], timeout=10)
            if client.health.health().status == 'ok':
                status = [0, f'(pgpt)']
            else:
                status = [2, client.health.health()]
        except Exception as err:
            status = [1, f'Сервер PrivateGPT не отвечает']
            app.logger.error(f'29: {err}')
    else:
        status = [1, f'Нет обработчика для {app.config["LLM"]} модели.']
    app.logger.info(f"return status: {app.config['LLM']}, {status}")
    # logging.info(f"return status: {app.config['LLM']}, {status}")
    return status


@app.before_request
def before_request():
    if current_user.is_authenticated:
        current_user.last_seen = datetime.now()
        db.session.commit()


@app.route('/test', methods=['GET', 'POST'])
@login_required
def test():
    return render_template('handl_answ.html')


@app.route('/product_files/<prdctid>', methods=['GET', 'POST'])
@login_required
def product_files(prdctid):
    cats = cat_pr_faq_f(prdctid)
    product = Products.query.filter_by(isact=1).filter_by(id=prdctid).first_or_404()
    files_db = Files.query.filter_by(isact=1).filter_by(prdct_id=prdctid)

    files = []
    for ef in files_db:
        nmb_faq = Faq.query.filter_by(file_id=ef.id).count()
        username = User.query.filter_by(id=Files.query.filter_by(isact=1).filter_by(id=ef.id)[0].wholoadfile)[
            0].username
        files.append([ef, dic_cat_file_f(ef.id), username, ef.tokens, ef.bathes, nmb_faq])
    return render_template('product_files_show.html', status=serv_status(), product=product,
                           files=files, cats=cats, brand=brand, brand_gpt=brand_gpt)


# Запрос GPT заполнить описание продукта перед сохранением этого описания в категории
@app.route('/askgpt', methods=['POST'])
@login_required
def askgpt():
    mess = request.json
    rsp = response_cat(mess)
    return jsonify(rsp)


# Запрос краткого описания продукта из формы index.html
@app.route('/getprshr', methods=['POST'])
@login_required
def getprshr():
    prd_id = int(request.json)
    cat_pr_faq = cat_pr_faq_f(prd_id)
    text = f''
    for ec in cat_pr_faq:
        text += f'{ec[1]}: {ec[7]}\n'
    return jsonify(text)


# Сохранение в faq описания категории продукта
@app.route('/svfaq', methods=['POST'])
@login_required
def svfaq():
    mess = request.json
    # print(mess)
    id, qu, an = mess['faq_id'], mess['quest'], mess['answ']
    print('id', id)
    faq = Faq.query.filter_by(id=id).one()
    if app.config['LLM'] == 'OpenAI':
        model = app.config['EMB_OPENAI_MODEL']
        client_oai = OpenAI(api_key=gl_api_key)
        emb_q_oai = pickle.dumps(client_oai.embeddings.create(model=model, input=qu).data[0].embedding)
        emb_a_oai = pickle.dumps(client_oai.embeddings.create(model=model, input=an).data[0].embedding)
        emb_q, emb_a = None, None
    elif app.config['LLM'] == 'PrivateGPT':
        client = PrivateGPTApi(base_url=app.config['URL_PGPT'], timeout=None)
        emb_q = pickle.dumps(client.embeddings.embeddings_generation(input=qu).data[0].embedding)
        emb_a = pickle.dumps(client.embeddings.embeddings_generation(input=an).data[0].embedding)
        emb_q_oai, emb_a_oai = None, None
    faq.question, faq.answer, faq.emb_q, faq.emb_a, faq.emb_q_oai, faq.emb_a_oai = qu, an, emb_q, emb_a, emb_q_oai, emb_a_oai
    db.session.commit()
    return jsonify({'error': '0'})


# Удалить файл из перечня в продукте
@app.route('/delete', methods=['POST'])
@login_required
def delete_file(fileid=None):
    client = PrivateGPTApi(base_url=app.config['URL_PGPT'], timeout=None)
    fileid = int(request.json)
    batch_lst = [eb.id for eb in Batch.query.filter_by(file_id=fileid)]
    row_to_delete = Files.query.get_or_404(fileid)
    # row_to_delete_batch = Batch.query.filter_by(file_id=fileid)
    prdctid = row_to_delete.prdct_id
    # Files.query.filter_by(isact=1).filter_by(id=fileid).delete()
    # Files.query.filter_by(isact=1).filter_by(id=fileid).isact = False
    row_to_delete.isact = False
    Batch.query.filter_by(file_id=fileid).delete()
    filename_to_delete = f'{app.config["UPLOAD_FOLDER"]}/{prdctid}/{row_to_delete.filename}'
    try:
        remove(filename_to_delete)
    except FileNotFoundError:
        app.logger.error(f'Нет файла для удаления {filename_to_delete}')
        # rsp = f'Нет файла для удаления {filename_to_delete}'
        # return jsonify({'error': '1', 'message': rsp})
    try:
        if app.config['LLM'] == 'OpenAI':
            pass
        elif app.config['LLM'] == 'PrivateGPT':
            client.ingestion.delete_ingested(row_to_delete.idfilegpt)
        else:
            rsp = f'Нет обработчика для выбранной LLM config.py {app.config["LLM"]} модели.'
            return jsonify({'error': '1', 'message': rsp})
    except Exception as err:
        rsp = f'Выбранная в LLM config.py {app.config["LLM"]} языковая модель недоступна.'
        app.logger.error(f'112: {err}')
        return jsonify({'error': '1', 'message': rsp})

    # client.ingestion.delete_ingested(row_to_delete.idfilegpt)
    db.session.commit()
    with db.engine.connect() as conn:
        conn.execute(delete(catgr_files).where(catgr_files.c.file_id == fileid))
        conn.execute(delete(catgr_batches).where(catgr_batches.c.batch_id.in_(batch_lst)))
        conn.commit()

    # проверить что в catgr_files catgr_batches удаление произошло
    rsp = f'Файл {filename_to_delete} удалён'
    # return redirect(url_for('product',prdctid=prdctid))
    return jsonify({'error': '0', 'message': rsp})


def allowed_file(filename):
    return '.'
    # return '.' in filename and \
    #     filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    client = PrivateGPTApi(base_url=app.config['URL_PGPT'], timeout=None)
    prdctid = int(request.form.get('prd_id'))
    product = Products.query.filter_by(isact=1).filter_by(id=prdctid).first_or_404()
    # print(dir(request.files))
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не найден в запросе'}), 400
    file = request.files['file']
    filename = file.filename
    ispublic = True if request.form.get('newpub') == '1' else False
    cats = Catgr.query.all()
    lst_cat = [ec.id for ec in cats if request.form.get(f'newcat-{ec.id}')]
    # print(lst_cat)
    if allowed_file(filename):
        filename_to_save = f'{app.config["UPLOAD_FOLDER"]}/{prdctid}/{filename}'
        dir_name = path.dirname(filename_to_save)
        makedirs(dir_name, exist_ok=True)
        file.save(filename_to_save)
        try:
            if app.config['LLM'] == 'OpenAI':
                ingested_file_doc_id = 0
            elif app.config['LLM'] == 'PrivateGPT':
                with open(filename_to_save, "rb") as f:
                    ingested_file_doc_id = client.ingestion.ingest_file(file=f).data[0].doc_id
            else:
                rsp = f'Нет обработчика для выбранной LLM config.py {app.config["LLM"]} модели.'
                ingested_file_doc_id = 0
                return jsonify({'error': '1', 'message': rsp})
        except Exception as err:
            rsp = f'Выбранная в LLM config.py {app.config["LLM"]} языковая модель недоступна.'
            app.logger.error(f'166: {err}')
            return jsonify({'error': '1', 'message': rsp})
        with open(filename_to_save, 'r') as f:
            text = f.read()
            file_hash = hash(text)
            tokens = num_tokens(text)
            bathes = split_strings_from_text(text)
        file = Files(filename=filename, filehash=file_hash, wholoadfile=int(current_user.id),
                     idfilegpt=ingested_file_doc_id, ispublic=ispublic, prdct_id=int(prdctid), tokens=tokens,
                     bathes=len(bathes))
        db.session.add(file)
        db.session.commit()
        with db.engine.connect() as conn:
            for ec in lst_cat:
                conn.execute(insert(catgr_files).values(cat_id=ec, file_id=file.id))
            conn.commit()
        split_file(file.id, bathes, lst_cat)
        access = 'публичного' if ispublic else 'ограниченного'
        flstr = f'Файл {filename} добавлен к описанию продукта {product.prdctname} в режиме {access} доступа.'
        if len(bathes) > 0:
            flstr = flstr + f'\nФайл составляет {tokens} токенов и разбит на {len(bathes)} частей.'
        return jsonify({'error': '0', 'message': flstr}), 200
    return jsonify({'error': 'Недопустимое имя файла или другая ошибка загрузки'}), 400


@app.route('/product/<prdctid>', methods=['GET', 'POST'])
@login_required
def product(prdctid):
    cats = cat_pr_faq_f(prdctid)
    # flash(f'Здесь будут появляться flash-сообщения')
    # prdctid = 2
    product = Products.query.filter_by(isact=1).filter_by(id=prdctid).first_or_404()
    # cats = Catgr.query.all()
    files_db = Files.query.filter_by(isact=1).filter_by(prdct_id=prdctid)
    files = []
    for ef in files_db:
        username = User.query.filter_by(id=Files.query.filter_by(isact=1).filter_by(id=ef.id)[0].wholoadfile)[
            0].username
        files.append([ef, dic_cat_file_f(ef.id), username, ef.tokens, ef.bathes])
    nh_answ = Answ_faq.query.filter_by(prdct_id=prdctid).filter_by(is_done=False).count()
    return render_template('product.html', status=serv_status(), product=product,
                           files=files, cats=cats, nh_answ=nh_answ, brand=brand, brand_gpt=brand_gpt)


@app.route('/prod_view/<prdctid>', methods=['GET', 'POST'])
@login_required
def prod_view(prdctid):
    product = Products.query.filter_by(isact=1).filter_by(id=prdctid).first_or_404()
    cats = cat_pr_faq_f(prdctid)
    return render_template('prod_view.html', status=serv_status(), product=product, cats=cats, brand=brand,
                           brand_gpt=brand_gpt)


# Отправка сообщений в чатбот из формы index.html
@app.route('/send', methods=['POST'])
@login_required
def send():
    mess = request.json
    rsp = response_json(current_user.id, mess)
    # print(rsp)
    # print(rsp['context'])
    return jsonify(rsp)


# Выбор темы из формы index.html
@app.route('/topic', methods=['POST'])
@login_required
def topic():
    topic = request.json
    rsp = topic_posts_f(current_user.id, topic)
    # print(rsp)
    return jsonify(rsp)


# index Начальная страница
@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    id_topics = topics_f(current_user.id)
    cats = Catgr.query.all()
    if serv_status()[0] == 1:
        flash(f'VPN + в личном кабинете введите действующий ключ Open AI API')
    products = Products.query.filter_by(isact=1).all()
    if current_user.rolepr_id in app.config['FULL_ACCESS_ROLE']:
        files = Files.query.filter_by(isact=1).all()
    else:
        files = Files.query.filter_by(isact=1).filter(Files.ispublic).all()
    form = PostForm()
    cntxstr = ast.literal_eval(current_user.cntxstr)
    prdct_ids, prdct_nms = prdct_id_nm(cntxstr)
    if form.validate_on_submit():
        cntxstr = {}
        for ef in files:
            if request.form.get(f'{ef.filehash}') is not None: cntxstr[str(ef.filehash)] = '1'
        str_cntxstr = str(cntxstr)
        post = Post(body=form.post.data, author=current_user, user_context=str_cntxstr)
        row_to_change = User.query.get_or_404(current_user.id)
        row_to_change.cntxstr = str_cntxstr
        db.session.add(post)
        db.session.commit()
        # postid = Post.query.filter(Post.user_id == current_user.id).order_by(Post.id.desc()).first().id
        postid = post.id
        # response(form.post.data,cntxstr,current_user.id, postid)
        return redirect(url_for('index'))
    return render_template("index.html", status=serv_status(), title=brand,
                           products=products, form=form, files=files, cntxstr=cntxstr,
                           prdct_nms=prdct_nms, id_topics=id_topics, cats=cats, brand=brand, brand_gpt=brand_gpt)


@app.route('/handl_answ', methods=['GET', 'POST'])
@app.route('/handl_answ/<prdctid>', methods=['GET', 'POST'])
@login_required
def handl_answ(prdctid=0):
    form = NewFAQ()
    form_id, form_name = "faq_add", "faq_add"
    posts = posts_to_view_to_handling(current_user.id, prdctid)
    prdcts = Products.query.filter_by(isact=1).all()
    # if len(posts) == 0:
    #     return render_template('handl_answ_0.html')

    if request.method == 'POST':
        lst_pst = []
        for ep in posts:
            if request.form.get(f'{ep[0]}-chk') is not None:
                lst_pst.append(ep[0])

        row_to_change = Post.query.filter(Post.id.in_(lst_pst))
        row_to_change = Answ_faq.query.filter(Answ_faq.id.in_(lst_pst))
        row_to_change.update({'is_done': True})
        question, answer = request.form.get(f'question'), request.form.get(f'answer')
        prd_id = request.form.get(f'product')
        prdctid = prd_id
        ispublic = True if request.form.get(f'ispublic') == 'y' else False
        if question is not None:
            try:
                if app.config['LLM'] == 'OpenAI':
                    model = app.config['EMB_OPENAI_MODEL']
                    client_oai = OpenAI(api_key=gl_api_key)
                    emb_q_oai = pickle.dumps(
                        client_oai.embeddings.create(model=model, input=question).data[0].embedding)
                    emb_a_oai = pickle.dumps(client_oai.embeddings.create(model=model, input=answer).data[0].embedding)
                    emb_q = None
                    emb_a = None
                elif app.config['LLM'] == 'PrivateGPT':
                    client = PrivateGPTApi(base_url=app.config['URL_PGPT'], timeout=None)
                    emb_q = pickle.dumps(client.embeddings.embeddings_generation(input=question).data[0].embedding)
                    emb_a = pickle.dumps(client.embeddings.embeddings_generation(input=answer).data[0].embedding)
                    emb_q_oai = None
                    emb_a_oai = None
                else:
                    emb_q, emb_a, emb_q_oai, emb_a_oai = None, None, None, None
                    app.logger.error(f'Нет обработчика для выбранной LLM config.py модели.')
            except Exception as err:
                emb_q, emb_a, emb_q_oai, emb_a_oai = None, None, None, None
                app.logger.error(f'Выбранная в LLM config.py языковая модель недоступна.')

            faq = Faq(question=question, answer=answer, emb_q=emb_q, emb_a=emb_a,
                      user_id=current_user.id, prdct_id=prd_id, ispublic=ispublic, emb_q_oai=emb_q_oai,
                      emb_a_oai=emb_a_oai)
            db.session.add(faq)
        db.session.commit()
        return render_template('handl_answ.html', status=serv_status(), posts=posts, form=form, id=form_id,
                               name=form_name, prdcts=prdcts, prdctid=prdctid, brand=brand, brand_gpt=brand_gpt)
    return render_template('handl_answ.html', status=serv_status(), posts=posts, form=form, id=form_id, name=form_name,
                           prdcts=prdcts, prdctid=prdctid, brand=brand, brand_gpt=brand_gpt)


@app.route('/test_register', methods=['GET', 'POST'])
def test_register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('test_register.html', title='Register', form=form, brand=brand, brand_gpt=brand_gpt)


@app.route('/test_login', methods=['GET', 'POST'])
def test_login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Неверное имя или пароль')
            return redirect(url_for('login'))
        login_user(user, remember=True)  # form.remember_me.data
        return redirect(url_for('index'))
    return render_template('test_login.html', title='Sign In', form=form, brand=brand, brand_gpt=brand_gpt)


@app.route('/products', methods=['GET', 'POST'])
@login_required
def products():
    form = ProductsForm()
    products = Products.query.filter_by(isact=1).all()

    if form.validate_on_submit():
        product = Products(prdctname=form.prdctname.data, mngr_id=form.manager.data)
        db.session.add(product)
        db.session.commit()
        p_id, u_id = product.id, current_user.id
        ques_shr = f'Опиши коротко, не более чем в трех предложениях, характеристику продукта по категории '
        ques_long = f'Опиши подробно характеристику продукта по категории '
        for ec in Catgr.query.all():
            faq = Faq(question=ques_shr, answer='', prdct_id=p_id, user_id=u_id, ispublic=True)
            db.session.add(faq)
            db.session.commit()
            faq_id_shr = faq.id
            faq = Faq(question=ques_long, answer='', prdct_id=p_id, user_id=u_id, ispublic=True)
            db.session.add(faq)
            db.session.commit()
            faq_id_long = faq.id
            with db.engine.connect() as conn:
                conn.execute(
                    insert(prd_cat_faq).values(prd_id=p_id, cat_id=ec.id, faq_id=faq_id_long, faq_shr_id=faq_id_shr))
                conn.commit()
        # flash(f'Вы ввели новый продукт {form.prdctname.data}!')
        return redirect(url_for('products'))
    return render_template("products.html", status=serv_status(), title='Products', form=form, products=products,
                           brand=brand, brand_gpt=brand_gpt)


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html', status=serv_status(), title='Register', form=form, brand=brand,
                           brand_gpt=brand_gpt)


@app.route('/user/<username>', methods=['GET', 'POST'])
@login_required
def user(username):  #
    if request.method == "POST":
        api = request.form.get(f'apikey')
        app.logger.info(f'пользователь {current_user} вел api_key {api}')
        if check_openai_api_key(api):
            set_openai_api_key(current_user, api)
        else:
            flash(f'Open AI API key не принят сервером, поробуйте другой')
    rols = rolepr.query.all()
    status = serv_status()
    return render_template('user.html', status=status, user=current_user, username=username, rols=rols, brand=brand,
                           brand_gpt=brand_gpt)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Неверное имя или пароль')
            return redirect(url_for('login'))
        login_user(user, remember=True)  # form.remember_me.data
        return redirect(url_for('index'))
    return render_template('login.html', status=serv_status(), title='Sign In', form=form, brand=brand,
                           brand_gpt=brand_gpt)


@app.route('/chngisp/<fileid>', methods=['GET', 'POST'])
@login_required
def chngisp(fileid):
    row_to_change = Files.query.filter_by(isact=1).get_or_404(fileid)
    prdctid = row_to_change.prdct_id
    ispublic = request.form.get(f'{fileid}')
    ispublic = True if ispublic == '1' else False
    row_to_change.ispublic = ispublic
    # row_to_change.ispublic = not row_to_change.ispublic
    db.session.commit()
    return redirect(url_for('product', prdctid=prdctid))


@app.route('/users', methods=['GET', 'POST'])
@login_required
def users():
    users = User.query.all()
    rols = rolepr.query.all()
    return render_template('users.html', status=serv_status(), users=users, rols=rols, brand=brand, brand_gpt=brand_gpt)


@app.route('/chngrl/<userid>', methods=['GET', 'POST'])
@login_required
def chngrl(userid):
    row_to_change = User.query.get_or_404(userid)
    rols = rolepr.query.all()
    selected_role = rols[row_to_change.rolepr_id]
    if request.method == "POST":
        selected_role = request.form.get(f'{row_to_change.username}')
        row_to_change.rolepr_id = selected_role
        db.session.commit()
        return redirect(url_for('users', userid=userid))


def result_context_oai(messages, cf_id):
    files = Files.query.filter_by(isact=1).filter(Files.id.in_(cf_id)).all()
    file_content = ''
    for ef in files:
        file_path = f'{app.config["UPLOAD_FOLDER"]}/{ef.prdct_id}/{ef.filename}'
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file_content + '\n' + file.read()
    new_messages = []
    for idx, em in enumerate(messages):
        if idx == (len(messages) - 1):
            new_mess = {"role": "user",
                        "content": f"Здесь контекст из файлов:\n{file_content}\n\nПожалуйста, ответь на вопрос: {em['content']}"}
        else:
            new_mess = em
        new_messages.append(new_mess)
    client_oai = OpenAI(api_key=gl_api_key)
    response = client_oai.chat.completions.create(
        messages=new_messages,
        model=app.config['OPENAI_MODEL']
    )
    app.logger.info(
        f'prompt_tokens: {response.usage.prompt_tokens}, completion_tokens: {response.usage.completion_tokens}')
    return response.choices[0].message.content.strip()


def result_no_context_oai(messages):
    client_oai = OpenAI(api_key=gl_api_key)
    response = client_oai.chat.completions.create(
        messages=messages,
        model=app.config['OPENAI_MODEL']
    )
    app.logger.info(
        f'prompt_tokens: {response.usage.prompt_tokens}, completion_tokens: {response.usage.completion_tokens}')
    return response.choices[0].message.content.strip()


# По id продкта возвращает список категорий и вопросов-ответов из FAQ, описывающих эти категории для данного прдукта
def cat_pr_faq_f(prd_id):
    cats = list(Catgr.query.all())
    with db.engine.connect() as conn:
        cat_faq = list(conn.execute(select(prd_cat_faq).where(prd_cat_faq.c.prd_id == prd_id)))
    cat_pr_faq = []
    for ec in cats:
        faq_id = [ecf[2] for ecf in cat_faq if ecf[1] == ec.id]
        if len(faq_id) != 0:
            if isinstance(faq_id[0], int):
                faq_id = faq_id[0]
                faq = Faq.query.filter_by(id=faq_id)[0]
                question = faq.question
                answer = faq.answer
            else:
                faq_id, question, answer = '', '', ''
        else:
            faq_id, question, answer = '', '', ''

        faq_id_shr = [ecf[3] for ecf in cat_faq if ecf[1] == ec.id]
        if len(faq_id_shr) != 0:
            if isinstance(faq_id_shr[0], int):
                faq_id_shr = faq_id_shr[0]
                faq = Faq.query.filter_by(id=faq_id_shr)[0]
                question_shr = faq.question
                answer_shr = faq.answer
            else:
                faq_id_shr, question_sht, answer_shr = '', '', ''
        else:
            faq_id_shr, question_shr, answer_shr = '', '', ''
        cat_pr_faq.append([ec.id, ec.name, faq_id, question, answer, faq_id_shr, question_shr, answer_shr])
    return cat_pr_faq


# По file_id возвращает список '1' и '0' - признаков есть такая категория у файла или нет (для формы Product.html)
def dic_cat_file_f(file_id):
    with db.engine.connect() as conn:
        cat_file = list(conn.execute(select(catgr_files).where(catgr_files.c.file_id == file_id)))
    cats_of_file = [ef[0] for ef in cat_file]
    cats = Catgr.query.order_by(Catgr.id).all()
    dic_cat_file = {}
    for ec in cats:
        dic_cat_file[ec.id] = '1' if ec.id in cats_of_file else '0'
    return dic_cat_file


# По выбранным в index.html контексам возвращает список id файлов для контекста запроса
def context_filter_id_f(context):
    # mess = {'gl_topic': '186', 'context': ['chkprd-1', 'chkprd-3', 'chkcat-2', 'chkfile-5', 'chkfile-6'], 'message': 'sfdf'}
    # context = mess['context']
    cat_lst, prd_lst, file_lst = [], [], []
    for ec in context:
        ecprt = ec.partition('-')
        if ecprt[0] == 'chkcat':
            cat_lst.append(int(ecprt[2]))
        elif ecprt[0] == 'chkprd':
            prd_lst.append(int(ecprt[2]))
        elif ecprt[0] == 'chkfile':
            file_lst.append(int(ecprt[2]))
    cntxt_str = {}
    if len(prd_lst): cntxt_str['prd'] = prd_lst
    if len(cat_lst): cntxt_str['cat'] = cat_lst
    if len(file_lst): cntxt_str['file'] = file_lst

    context_filter_file = []
    for ef in Files.query.filter_by(isact=1).filter(Files.id.in_(file_lst)).all():
        if ef.id not in context_filter_file:
            context_filter_file.append(ef.id)
    context_filter_prd = []
    for ef in Files.query.filter_by(isact=1).filter(Files.prdct_id.in_(prd_lst)).all():
        if ef.id not in context_filter_prd:
            context_filter_prd.append(ef.id)
    context_filter_id = context_filter_file
    for ef in context_filter_prd:
        if ef not in context_filter_id:
            ec_cat_sel = select(catgr_files).where(catgr_files.c.file_id == ef)
            with db.engine.connect() as conn:
                for ecf in conn.execute(ec_cat_sel):
                    if ecf[0] in cat_lst:
                        context_filter_id.append(ef)
                        break
    return context_filter_id, cntxt_str, prd_lst


# По id файлов возвращает список id-PrivateGPT файлов для контекста запроса
def context_filter_f(context_filter_id):
    context_filter = []
    files = Files.query.filter_by(isact=1).filter(Files.id.in_(context_filter_id)).all()
    for ef in files:
        context_filter.append(ef.idfilegpt)
    return context_filter


# Сборка текста запроса из последнего сообщения и предыдущих из текущей темы
def collect_mess(user_id, topic, message):
    # user_id = 4
    # topic = 1
    sys_prompt = f'Ответь на русском языке'
    messages = [{"role": "user", "content": message}]
    if topic:
        pr_post = Topic.query.filter_by(id=topic)[0].post_id
        while pr_post:
            post = Post.query.filter_by(id=pr_post)[0]
            if post.user_id == 1:
                messages.insert(0, {"role": "assistant", "content": post.body})
            if post.user_id == user_id:
                messages.insert(0, {"role": "user", "content": post.body})
            pr_post = post.reply_id
    messages.insert(0, {"role": "system", "content": sys_prompt})
    return messages


# Пока не используется. Контроль длины запроса (вместиться ли в контекстное окно) и подготовка запроса.
def check_context_window_f(mess):
    if mess['topic']:
        int(mess['topic'])
    message = '9999'
    # ------
    return message


# Обработчик запросов к чатботу для заполнения описания продукта
def response_cat(mess):
    sys_prompt = f'Ответь на русском языке'
    messages = [{"role": "system", "content": sys_prompt}]
    messages.append({"role": "user", "content": mess['message']})
    if mess['context']:
        file_lst = []
        for ec in mess['context']:
            ecprt = ec.partition('-')
            if ecprt[0] == 'chkfile':
                file_lst.append(int(ecprt[2]))
        context_filter = context_filter_f(file_lst)
        try:
            if app.config['LLM'] == 'OpenAI':
                result = result_context_oai(messages, file_lst)
            elif app.config['LLM'] == 'PrivateGPT':
                result = result_context(messages, context_filter)
            else:
                result = f'Нет обработчика для выбранной LLM config.py модели.'
        except Exception as err:
            result = f'Выбранная в LLM config.py языковая модель недоступна.'
            app.logger.error(f"170 :{err}")
    else:
        try:
            if app.config['LLM'] == 'OpenAI':
                result = result_no_context_oai(messages)
            elif app.config['LLM'] == 'PrivateGPT':
                result = result_no_context(messages)
            else:
                result = f'Нет обработчика для выбранной LLM config.py модели.'
        except Exception as err:
            result = f'Выбранная в LLM config.py языковая модель недоступна.'
    return result


# Главный обработчик запросов к чатботу
def response_json(user_id, mess):
    answ_faq = None
    post_fu = Post(body=mess['message'], user_id=user_id)
    topic = mess['topic']
    if topic: post_fu.reply_id = Topic.query.filter_by(id=topic)[0].post_id
    messages = collect_mess(user_id, topic, mess['message'])
    if mess['context']:
        cf_id, cntxt_str, prd_lst = context_filter_id_f(mess['context'])
        post_fu.user_context = str(cntxt_str)
        context_filter = context_filter_f(cf_id)
        try:
            if app.config['LLM'] == 'OpenAI':
                result = result_context_oai(messages, cf_id)
            elif app.config['LLM'] == 'PrivateGPT':
                result = result_context(messages, context_filter)
            else:
                result = f'Нет обработчика для выбранной LLM config.py модели.'
        except Exception as err:
            result = f'Выбранная в LLM config.py языковая модель недоступна (VPN?).'
        cf_id = str(cf_id)

    else:
        cf_id, prd_lst = None, None
        try:
            if app.config['LLM'] == 'OpenAI':
                result = result_no_context_oai(messages)
            elif app.config['LLM'] == 'PrivateGPT':
                result = result_no_context(messages)
            else:
                result = f'Нет обработчика для выбранной LLM config.py модели.'
        except Exception as err:
            result = f'Выбранная в LLM config.py языковая модель недоступна.'
            app.logger.error(f'218: {err}')
    # check_context_window_f(messages, context_filter)
    # try:
    # print("start 217")
    db.session.add(post_fu)
    db.session.commit()
    post_fu_id = post_fu.id
    if topic:
        db_topic = Topic.query.filter_by(id=topic)[0]
    else:
        db_topic = Topic(text=mess['message'][:64], user_id=user_id)
        db.session.add(db_topic)
        db.session.commit()
        topic = db_topic.id  # , mess['message'][:64]]
    post_fu.topic = topic
    post = Post(body=f'{result}', user_id=1, reply_id=post_fu_id, user_context=cf_id, is_done=False, topic=topic)
    db.session.add(post)
    db.session.commit()
    db_topic.post_id = post.id
    db.session.commit()
    # print("start 234", prd_lst, user_id, post_fu_id)
    # print("start 235", prd_lst)
    answ_faq = Answ_faq_f(mess['message'], prd_lst, user_id, post_fu_id) if prd_lst else None
    # except Exception as e:
    #     print(f'Ошибка вот здесь {e}')

    # return result, topic, answ_faq if answ_faq else result, topic
    if answ_faq is not None:
        return result, topic, answ_faq
    else:
        return result, topic


# Устарело?
def prdct_id_nm(cntxstr):
    files = Files.query.filter_by(isact=1).all()
    prdct_ids, prdct_nms = [], []
    for ef in files:
        if str(ef.filehash) in cntxstr.keys():
            prdct_name = Products.query.filter_by(isact=1).filter(Products.id == ef.prdct_id).first().prdctname
            prdct_id = Products.query.filter_by(isact=1).filter(Products.id == ef.prdct_id).first().id
            if prdct_id not in prdct_ids:
                prdct_ids.append(prdct_id)
                prdct_nms.append(prdct_name)
    return prdct_ids, prdct_nms


# Строка контекста для отправки в index.html при выборе темы (./topic)
def context_lst_f(post_id):
    context_str = Post.query.filter_by(id=post_id)[0].user_context
    context = ast.literal_eval(context_str)
    context_lst = []
    for ep in context:
        epdic = {}
        epdic['id'] = ep[0]
        epdic['prdctname'] = Products.query.filter_by(isact=1).filter_by(id=ep[0])[0].prdctname
        file_lst = []
        for ef in ep[1]:
            efdic = {}
            efdic['filename'] = Files.query.filter_by(isact=1).filter_by(isact=1).filter_by(filehash=ef)[0].filename
            efdic['filehash'] = ef
            file_lst.append(efdic)
            # file_lst.append([ef, Files.query.filter_by(isact=1).filter_by(filehash = ef)[0].filename])
            # print(ef)
        epdic['files'] = file_lst
        context_lst.append(epdic)
    return context_lst


# Права доступа по id пользователя
def is_all(user_id):
    full_acces = True if User.query.filter(User.id == user_id).first().rolepr_id in app.config[
        'FULL_ACCESS_ROLE'] else False
    return full_acces


# Инициация dataframe для обработки совпадений FAQ
def df_init(prd_id, user_id):
    if is_all(user_id):
        faqs = Faq.query.filter_by(prdct_id=prd_id).all()
    else:
        faqs = Faq.query.filter_by(prdct_id=prd_id).filter_by(ispublic=True).all()
    df = pd.DataFrame([eq.__dict__ for eq in faqs])
    if df.shape[0] > 0:
        df = df.drop('_sa_instance_state', axis=1)
    return df


# top_n первых ответа из FAQ с высшим рейтингом
def strings_ranked_by_relatedness(
        query: str,
        df: pd.DataFrame,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = app.config['NUMBERS_FAQ_REPLY']  # 1 #
) -> tuple[list[str], list[float]]:
    if df.shape[0] > 0:
        # try:
        if app.config['LLM'] == 'OpenAI':
            """Returns a list of strings and relatednesses, sorted from most related to least."""
            client_oai = OpenAI(api_key=gl_api_key)
            openai_model = app.config['EMB_OPENAI_MODEL']
            query_embedding_response = client_oai.embeddings.create(
                model=openai_model,
                input=query,
            )
            query_embedding = query_embedding_response.data[0].embedding
            strings_and_relatednesses = [
                (row["id"], row["question"], relatedness_fn(query_embedding, pickle.loads(row["emb_q_oai"])))
                for i, row in df.iterrows()
            ]
            strings_and_relatednesses.sort(key=lambda x: x[2], reverse=True)
            id, strings, relatednesses = zip(*strings_and_relatednesses)
            return id[:top_n], strings[:top_n], round(relatednesses[:top_n], 2)

        elif app.config['LLM'] == 'PrivateGPT':
            client = PrivateGPTApi(base_url=app.config['URL_PGPT'], timeout=None)
            """Returns a list of strings and relatednesses, sorted from most related to least."""
            embedding_result = client.embeddings.embeddings_generation(input=query)
            query_embedding = embedding_result.data[0].embedding
            strings_and_relatednesses = [
                (row["id"], row["question"], relatedness_fn(query_embedding, pickle.loads(row["emb_q"])))
                for i, row in df.iterrows()
            ]
            strings_and_relatednesses.sort(key=lambda x: x[2], reverse=True)
            id, strings, relatednesses = zip(*strings_and_relatednesses)
            return id[:top_n], strings[:top_n], round(relatednesses[:top_n], 2)
        else:
            app.logger.error(f'Нет обработчика для выбранной LLM config.py модели')
            return [], [], []
        # except Exception as err:
        app.logger.error(f'Выбранная в LLM config.py языковая модель недоступна, {err}')
        return [], [], []
    else:
        app.logger.error(f'Нет записей FAQ')
        return [], [], []


# Выборка необработанных для FAQ постов (запрос и отправки из формы handl_answ.html)
def posts_to_view_to_handling(user_id, prdctid):
    # prd = Products.query.filter_by(mngr_id=user_id).all()
    quest_answ, lst_apost = [], []
    if prdctid == 0:
        prd_lst = [epr.id for epr in Products.query.filter_by(isact=1).filter_by(mngr_id=user_id).all()]
    else:
        prd_lst = [epr.id for epr in
                   Products.query.filter_by(isact=1).filter_by(id=prdctid).filter_by(mngr_id=user_id).all()]
    aposts = Answ_faq.query.filter(and_(Answ_faq.prdct_id.in_(prd_lst), Answ_faq.is_done != True)).order_by(
        Answ_faq.prdct_id, Answ_faq.id_quest.desc(), Answ_faq.id).all()
    for ep in aposts:
        if ep.id_quest not in lst_apost:
            quest = f'{Post.query.filter_by(id=ep.id_quest)[0].body}'
            answer = Post.query.filter_by(reply_id=ep.id_quest)[0].body
            faq_quest = Faq.query.filter_by(id=ep.id_faq)[0].question
            faq_answ = Faq.query.filter_by(id=ep.id_faq)[0].answer
            faq = f'FAQ({ep.id_faq}:{ep.rltdns}): {faq_quest} | {faq_answ} )'
            quest_answ.append([ep.id, quest, answer, faq])
            # lst_apost.append(ep.id_quest) #Если ответов из FAQ больше, чем 1, но надо ограничить одним
    return quest_answ


# Запрос в PrivateGPT
def result_context(text, context_filter):
    client = PrivateGPTApi(base_url=app.config['URL_PGPT'], timeout=None)
    result = client.contextual_completions.chat_completion(
        messages=text,
        use_context=True,
        context_filter={"docs_ids": context_filter},
        include_sources=True,
    ).choices[0].message.content.strip()
    return result


# Запрос в PrivateGPT
def result_no_context(text):
    client = PrivateGPTApi(base_url=app.config['URL_PGPT'], timeout=None)
    result = client.contextual_completions.chat_completion(
        messages=text,
        use_context=False,
    ).choices[0].message.content.strip()
    return result


# Выборка ответов из FAQ для отображения после ответа ChatGPT
def Answ_faq_f(text, prdct_ids, user_id, postid):
    # prdct_ids , prdct_nms = prdct_id_nm(cntxstr)
    answ_faq = []
    for ep in prdct_ids:
        id, reply_text, rlt = strings_ranked_by_relatedness(text, df_init(ep, user_id))
        for i, eid in enumerate(id):
            answ_text = Faq.query.filter_by(id=eid)[0].answer
            answ_faq.append([ep, rlt[i], answ_text])
            answ_f = Answ_faq(id_quest=postid, id_faq=eid, rltdns=rlt[i], prdct_id=ep)
            db.session.add(answ_f)
            db.session.commit()
    return answ_faq


# Выборка тем для пользователя для раздела навигации в index.html
def topics_f(user_id):
    topics = Topic.query.filter_by(user_id=user_id).order_by(Topic.post_id.desc()).limit(7)
    id_topics = []
    for et in topics:
        id_topics.append([et.id, et.text[:27] + '...'])
    return id_topics


def topic_posts_f(user_id, topic):
    topic_posts, context = [], None
    if topic:
        pr_post = Topic.query.filter_by(id=topic)[0].post_id
        context = Post.query.filter_by(id=Post.query.filter_by(id=pr_post)[0].reply_id)[0].user_context
        while pr_post:
            post = Post.query.filter_by(id=pr_post)[0]
            if post.user_id == 1:
                topic_posts.insert(0, {'assistant': post.body})
            if post.user_id == user_id:
                topic_posts.insert(0, {'user': post.body})
            pr_post = post.reply_id
    if context:
        return {'topic_posts': topic_posts, 'context': ast.literal_eval(context)}
    else:
        return {'topic_posts': topic_posts}


# -----------------------
# Ранее был файл files.py
tokenizer = MistralTokenizer.v1()
max_tokens = app.config['MAX_TOKENS_IN_BATCH']
delimiters = app.config['DELIMITERS']


# client = PrivateGPTApi(base_url=app.config['URL_PGPT'], timeout=None)

# Подсчитывает количество токенов в строке
def num_tokens(text: str):
    try:
        if app.config['LLM'] == 'OpenAI':
            encoding = tiktoken.encoding_for_model(app.config['OPENAI_MODEL'])
            tokens = len(encoding.encode(text))
        elif app.config['LLM'] == 'PrivateGPT':
            completion_request = ChatCompletionRequest(messages=[UserMessage(content=text)])
            tokens = len(tokenizer.encode_chat_completion(completion_request).tokens)
        else:
            app.logger.error(f'Нет обработчика для выбранной LLM config.py модели.')
            # количество слов
            tokens = len(text.split())
    except Exception as err:
        app.logger.error(f'Выбранная в LLM config.py языковая модель недоступна.')
        # количество слов
        tokens = len(text.split())
    return tokens


# Подсчитывает количество токенов в сообщнеии
def tot_tokens(messages: list):
    total_tokens = 0
    for message in messages:
        total_tokens += num_tokens(message["role"])
        total_tokens += num_tokens(message["content"])
    return total_tokens


# Выделяет из текста (строки) первую чать, которая не больше max_tokens
def trunc_string(string, max_tokens):
    if num_tokens(string) <= max_tokens:
        truncated_string = string
    else:
        try:
            if app.config['LLM'] == 'OpenAI':
                encoding = tiktoken.encoding_for_model(app.config['OPENAI_MODEL'])
                tokens = encoding.encode(string)
                truncated_tokens = tokens[:max_tokens]
                truncated_string = encoding.decode(truncated_tokens)
            elif app.config['LLM'] == 'PrivateGPT':
                completion_request = ChatCompletionRequest(messages=[UserMessage(content=string)])
                tokens = tokenizer.encode_chat_completion(completion_request).tokens
                truncated_string = tokenizer.decode(tokens[:max_tokens]).partition('[INST] ')[2].partition(' [/INST]')[
                    0].lstrip().rstrip()
            else:
                app.logger.error(f'Нет обработчика для выбранной LLM config.py модели.')
                truncated_string = string
        except Exception as err:
            app.logger.error(f'Выбранная в LLM config.py языковая модель недоступна.')
            truncated_string = string
    return truncated_string


# Рекурсивно разделяет текст (строку) по найденному разделителю с учетом того,
# что длина первой части должна быть меньше max_tokens
def del_string(string, max_tokens, delimiters):
    if num_tokens(string) <= max_tokens:
        left_str_out, right_str = string, ''
    else:
        truncated_string = trunc_string(string, max_tokens)
        for delimiter in delimiters:
            crt_str = truncated_string.rpartition(delimiter)
            dlm = delimiter
            left_str, right_str = crt_str[0], crt_str[2]
            if left_str == '' and delimiter == delimiters[-1]:
                left_str, right_str, dlm = crt_str[2], '', ''
            elif left_str == '':
                continue
            else:
                break
        rst, left_str_out, right_str = string.partition(left_str + dlm)
    return left_str_out.lstrip().rstrip(), right_str.lstrip().rstrip()


# Разбивает text на блоки (в списке str_list) размером не более чем по max_tokens
def split_strings_from_text(text, max_tokens=max_tokens, delimiters=delimiters):
    string = text
    num_tokens_in_string = num_tokens(string)
    if num_tokens_in_string <= max_tokens:
        return [string]
    else:
        left_str, right_str = del_string(string, max_tokens, delimiters)
        str_list = [left_str]
        while len(right_str) > 0:
            # print('--',right_str,'==')
            left_str, right_str = del_string(right_str, max_tokens, delimiters)
            str_list.append(left_str)
    return str_list


# Записывает разбитый на блоки текст из str_list в базу данных вместе с embeddings частей
def split_file(file_id: object, str_list: object, lst_cat) -> object:
    for estr in str_list:
        try:
            if app.config['LLM'] == 'OpenAI':
                client_oai = OpenAI(api_key=gl_api_key)
                openai_model = app.config['EMB_OPENAI_MODEL']
                query_embedding_response = client_oai.embeddings.create(
                    model=openai_model,
                    input=estr,
                )
                embed_oai = pickle.dumps(query_embedding_response.data[0].embedding)
                embed = None
            elif app.config['LLM'] == 'PrivateGPT':
                embed_oai = None
                client = PrivateGPTApi(base_url=app.config['URL_PGPT'], timeout=None)
                embed = pickle.dumps(client.embeddings.embeddings_generation(input=estr).data[0].embedding)
            else:
                app.logger.error(f'Нет обработчика для выбранной LLM config.py модели.')
                embed, embed_oai = None, None
        except Exception as err:
            app.logger.error(f'Выбранная в LLM config.py языковая модель недоступна (VPN?). {err}')
            embed, embed_oai = None, None
        batch = Batch(text=estr, embed=embed, embed_oai=embed_oai, file_id=file_id)
        db.session.add(batch)
        db.session.commit()
        with db.engine.connect() as conn:
            for ec in lst_cat:
                conn.execute(insert(catgr_batches).values(cat_id=ec, batch_id=batch.id))
            conn.commit()
    return str_list


def create_FAQ_by_file(prdid, file_id):
    prd = Products.query.filter_by(id=prdid).one()
    files = Files.query.filter_by(isact=1).filter_by(id=file_id).all()
    file_content = ''
    for ef in files:
        file_path = f'{app.config["UPLOAD_FOLDER"]}/{prdid}/{ef.filename}'
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file_content + '\n' + file.read()
    faqs = Faq.query.filter_by(file_id=file_id).all()
    faqs_l = []
    for eq in faqs:
        faqs_l.append(eq.question)

    class QuestionAnswer(BaseModel):
        question: str
        answer: str

    class QuestAnswList(BaseModel):
        qwanswlist: list[QuestionAnswer]

    sys_prompt = (f'Ты помогаешь составить базу данных вопросов и ответов по теме {prd.prdctname}.'
                  f'Информация о теме в тройных кавычках """{file_content}""".'
                  f'В базе данных уже есть вопросы, они перечислены между квадратными скобками {faqs_l}'
                  f'Сформулируй список вопросов и ответов на них, не повторяя вопросы, которые уже есть в списке.')
    messages = [{"role": "system", "content": sys_prompt}]
    client_oai = OpenAI(api_key=gl_api_key)
    openai_model = app.config['OPENAI_MODEL']
    if check_openai_api_key(gl_api_key):
        response = client_oai.beta.chat.completions.parse(
            messages=messages,
            model=openai_model,
            response_format=QuestAnswList,
            max_tokens=5000,
        )
        app.logger.info(
            f'prompt_tokens: {response.usage.prompt_tokens}, completion_tokens: {response.usage.completion_tokens}')
        completion = response.choices[0].message.parsed
        return True, completion.qwanswlist
    else:
        return False, None


@app.route('/faqfile', methods=['POST'])
@login_required
def save_faq_from_file():
    fileid = int(request.json)
    file = Files.query.filter_by(isact=1).filter_by(id=fileid).one()
    prd = Products.query.filter_by(isact=1).filter_by(id=file.prdct_id).one()
    isworkd, qwanswl = create_FAQ_by_file(prd.id, fileid)
    if isworkd:
        for eqa in qwanswl:
            try:
                if app.config['LLM'] == 'OpenAI':
                    model = app.config['EMB_OPENAI_MODEL']
                    client_oai = OpenAI(api_key=gl_api_key)
                    emb_q_oai = pickle.dumps(
                        client_oai.embeddings.create(model=model, input=eqa.question).data[0].embedding)
                    emb_a_oai = pickle.dumps(
                        client_oai.embeddings.create(model=model, input=eqa.answer).data[0].embedding)
                    emb_q = None
                    emb_a = None
                elif app.config['LLM'] == 'PrivateGPT':
                    client = PrivateGPTApi(base_url=app.config['URL_PGPT'], timeout=None)
                    emb_q = pickle.dumps(client.embeddings.embeddings_generation(input=eqa.question).data[0].embedding)
                    emb_a = pickle.dumps(client.embeddings.embeddings_generation(input=eqa.answer).data[0].embedding)
                    emb_q_oai = None
                    emb_a_oai = None
                else:
                    emb_q, emb_a, emb_q_oai, emb_a_oai = None, None, None, None
                    app.logger.error(f'Нет обработчика для выбранной LLM config.py модели.')
            except Exception as err:
                emb_q, emb_a, emb_q_oai, emb_a_oai = None, None, None, None
                app.logger.error(f'Выбранная в LLM config.py языковая модель недоступна.')

            app.logger.info(
                f'Вопрос {eqa.question} доавлен к описанию продукта {prd.prdctname} по файлу {file.filename}')
            faq = Faq(question=eqa.question, answer=eqa.answer, emb_q=emb_q, emb_a=emb_a,
                      user_id=current_user.id, prdct_id=prd.id, ispublic=file.ispublic, emb_q_oai=emb_q_oai,
                      emb_a_oai=emb_a_oai, file_id=fileid, isgptauto=1)
            db.session.add(faq)
            db.session.commit()
        rsp = f'По файлу {file.filename} составлен FAQ'
        return jsonify({'error': '0', 'message': rsp})
    else:
        app.logger.error(f'По файлу {file.filename} не составлен FAQ, нет ответа от LLM')
        rsp = f'LLM недоступна, по файлу {file.filename} не составлен FAQ'
        return jsonify({'error': '1', 'message': rsp})


# Удалить файл из перечня в продукте
@app.route('/del_product', methods=['POST'])
@login_required
def delete_product():
    try:
        prdid = int(request.json)
        row_to_delete = Products.query.get_or_404(prdid)
        files = Files.query.filter_by(isact=1).filter_by(prdct_id=prdid).all()
        for ef in files:
            delete_file(fileid=ef.id)
        row_to_delete.isact = False
        db.session.commit()
        rsp = f'Продукт {row_to_delete.prdctname} удален'
        app.logger.info(f'Продукт {row_to_delete.prdctname} удален пользователем {current_user}')
        return jsonify({'error': '0', 'message': rsp})
    except Exception as err:
        app.logger.error(f'Продукт {row_to_delete.prdctname} не удален, Error: {err}')
        rsp = f'Продукт {row_to_delete.prdctname} не удален, причина в логах.'
        return jsonify({'error': '1', 'message': rsp})


@app.route('/faq', methods=['GET', 'POST'])
@app.route('/faq/<prdctid>', methods=['GET', 'POST'])
@login_required
def faq(prdctid=None):
    ishide = request.args.get('hide','true') == 'true'
    form = EditFAQ()
    form_id, form_name = "faq_add", "faq_add"
    prdcts = Products.query.filter_by(isact=1).all()
    if prdctid:
        prdcts_faq = Products.query.filter_by(isact=1).filter_by(id=prdctid).all()
    else:
        prdcts_faq = Products.query.filter_by(isact=1).all()
    product_ids = [product.id for product in prdcts_faq]  # Извлекаем id активных продуктов
    if ishide:
        faqs = Faq.query.filter(Faq.prdct_id.in_(product_ids)).filter_by(isverified=False).all()
    else:
        faqs = Faq.query.filter(Faq.prdct_id.in_(product_ids)).all()
    if request.method == 'POST':
        lst_faq = []
        for ef in faqs:
            if request.form.get(f'{ef.id}-chk') is not None:
                lst_faq.append(ef.id)
        row_to_change = Faq.query.filter(Faq.id.in_(lst_faq)) # или .all()?
        row_to_change.update({'isverified': True})

        idfaq, question, answer = request.form.get(f'idfaq'), request.form.get(f'question'), request.form.get(f'answer')
        prd_id = request.form.get(f'product')
        prdctid = prd_id
        ispublic = True if request.form.get(f'ispublic') == 'y' else False
        if question is not None:
            try:
                if app.config['LLM'] == 'OpenAI':
                    model = app.config['EMB_OPENAI_MODEL']
                    client_oai = OpenAI(api_key=gl_api_key)
                    emb_q_oai = pickle.dumps(
                        client_oai.embeddings.create(model=model, input=question).data[0].embedding)
                    emb_a_oai = pickle.dumps(client_oai.embeddings.create(model=model, input=answer).data[0].embedding)
                    emb_q = None
                    emb_a = None
                elif app.config['LLM'] == 'PrivateGPT':
                    client = PrivateGPTApi(base_url=app.config['URL_PGPT'], timeout=None)
                    emb_q = pickle.dumps(client.embeddings.embeddings_generation(input=question).data[0].embedding)
                    emb_a = pickle.dumps(client.embeddings.embeddings_generation(input=answer).data[0].embedding)
                    emb_q_oai = None
                    emb_a_oai = None
                else:
                    emb_q, emb_a, emb_q_oai, emb_a_oai = None, None, None, None
                    app.logger.error(f'Нет обработчика для выбранной LLM config.py модели.')
            except Exception as err:
                emb_q, emb_a, emb_q_oai, emb_a_oai = None, None, None, None
                app.logger.error(f'Выбранная в LLM config.py языковая модель недоступна.')
            row_to_change = Faq.query.filter_by(id=idfaq)
            row_to_change.update(dict(question=question, answer=answer, emb_q=emb_q, emb_a=emb_a,
                      user_id=current_user.id, prdct_id=prd_id, ispublic=ispublic, emb_q_oai=emb_q_oai,
                      emb_a_oai=emb_a_oai))
        db.session.commit()
        return render_template('faq.html', status=serv_status(), faqs=faqs, form=form, id=form_id,
                               name=form_name, prdcts=prdcts, prdctid=prdctid, brand=brand, brand_gpt=brand_gpt, ishide=ishide)
    return render_template('faq.html', status=serv_status(), faqs=faqs, form=form, id=form_id, name=form_name,
                           prdcts=prdcts, prdctid=prdctid, brand=brand, brand_gpt=brand_gpt, ishide=ishide)
