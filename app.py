import streamlit as st
import json, random, io
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

st.set_page_config(page_title='문학 인물 인터뷰', page_icon='📖', layout='centered')
DATA=Path('data/works.json'); DATA.parent.mkdir(exist_ok=True)
ADMIN_PASSWORD='6460'; MODEL='gpt-6-luna'

def load():
    try: return json.loads(DATA.read_text(encoding='utf-8')) if DATA.exists() else {}
    except: return {}
def save(x): DATA.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def client():
    try: key=st.secrets['OPENAI_API_KEY']
    except: key=None
    return OpenAI(api_key=key) if key else None
def ai(instructions, prompt):
    c=client()
    if not c: raise RuntimeError('OPENAI_API_KEY가 없습니다. Streamlit Cloud의 Secrets에 API 키를 입력하세요.')
    return c.responses.create(model=MODEL,instructions=instructions,input=prompt).output_text.strip()
def ctx(w,cname):
    c=next((x for x in w.get('characters',[]) if x['name']==cname),{})
    return f'''[작품] {w.get('title','')}\n[작가] {w.get('author','')}\n[본문]\n{w.get('text','')}\n[교사 참고 자료]\n{w.get('reference','')}\n[등장인물]\n{cname}\n[성격/말투]\n{c.get('traits','')}'''
def hist():
    return '\n'.join(('학생: ' if x['role']=='student' else '인물: ')+x['content'] for x in st.session_state.get('messages',[]))
def normal(w,c,q):
    return ai(f'''너는 소설 속 등장인물 {c}이다.\n{ctx(w,c)}\n학생의 질문에 인물의 입장에서 답한다. 사실은 1순위 본문, 2순위 교사 자료, 3순위 본문의 행동·관계에 따른 합리적 추론 순으로 사용한다. 작품 밖 지식을 사실처럼 만들지 않는다. 인물의 성격과 말투를 반영한다. 욕설·비속어·성적·혐오 표현을 쓰지 않는다. 정답이나 교사의 해설을 직접 알려주지 않는다. 질문이 너무 짧으면 자연스럽게 구체화하도록 유도한다. 이전 대화:\n{hist()}''',q)
def make_mistake(w,c,q):
    instruction=f'''너는 소설 속 등장인물 {c}이다.\n{ctx(w,c)}\n이번 답변은 학생이 본문을 다시 읽어야 찾을 수 있는 '의도된 오답'이다. 학생 질문에 직접 답하되, 본문에 실제 존재하는 사건·인물·관계를 바탕으로 원인과 결과, 사건의 선후, 행동 동기, 관계 또는 사실 중 하나를 교묘하게 왜곡한다. 단순한 황당한 거짓말이나 오타는 금지한다. 학생이 본문에서 구체적 근거를 찾아 반박할 수 있어야 한다. 인물의 말투는 유지한다. 이전 대화:\n{hist()}'''
    for _ in range(3):
        a=ai(instruction,q)
        check=ai('너는 문학 교육용 오답 검증자다. JSON만 출력한다.',f'''본문:\n{w.get('text','')}\n질문:{q}\n답변:{a}\n\n다음 형식으로만 답해라: {{"valid":true/false,"reason":"이유"}}. valid는 (1)질문과 관련되고 (2)본문과 실제로 충돌하며 (3)학생이 본문에서 구체적으로 반박할 수 있고 (4)너무 쉽게 드러나는 오류가 아닐 때만 true다.''')
        try:
            check=check.replace('```json','').replace('```','').strip(); ok=json.loads(check).get('valid',False)
        except: ok=False
        if ok: return a
    raise RuntimeError('의도된 오답의 교육적 품질을 검증하지 못했습니다. 다시 시도해 주세요.')
def start(title,c):
    st.session_state.update(page='student',selected_work=title,character=c,current_turn=0,messages=[],mistake_turn=random.randint(1,5),mistake_used=False,finished=False)
def result_png(title,c,name,msgs,guess,evidence,judgment):
    img=Image.new('RGB',(1200,1500),'white'); d=ImageDraw.Draw(img)
    paths=['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc','C:/Windows/Fonts/malgun.ttf']; fp=next((p for p in paths if Path(p).exists()),None)
    f=ImageFont.truetype(fp,24) if fp else ImageFont.load_default(); h=ImageFont.truetype(fp,32) if fp else f
    y=40; d.text((50,y),'문학 인물 인터뷰 결과',font=h,fill='black'); y+=55
    for line in [f'작품: {title}',f'등장인물: {c}',f'학생: {name or "익명"}']:
        d.text((50,y),line,font=f,fill='black'); y+=38
    y+=15
    for i,m in enumerate(msgs,1):
        text=f'{i}. {"나" if m["role"]=="student" else c}: {m["content"]}'
        d.text((50,y),text[:100],font=f,fill='black'); y+=35
        if y>1050: break
    y=1100; d.text((50,y),f'내가 찾은 오답: {guess}번째 답변 / 판정: {judgment}',font=f,fill='black'); y+=45
    d.text((50,y),'본문 근거:',font=f,fill='black'); y+=35; d.text((50,y),evidence[:160],font=f,fill='black')
    b=io.BytesIO(); img.save(b,'PNG'); return b.getvalue()

def admin():
    st.title('⚙️ 교사용 관리자 화면'); pw=st.text_input('관리자 비밀번호',type='password')
    if pw!=ADMIN_PASSWORD: st.info('관리자 비밀번호를 입력하세요.'); return
    works=load(); st.success('관리자 화면에 들어왔습니다.')
    with st.form('work'):
        title=st.text_input('소설 제목'); author=st.text_input('작가'); text=st.text_area('소설 본문',height=300); ref=st.text_area('교사 참고 자료',height=180)
        n=st.number_input('등장인물 수',1,20,3); chars=[]
        for i in range(int(n)):
            a,b=st.columns([1,2]); name=a.text_input(f'{i+1}번 인물 이름',key=f'n{i}'); traits=b.text_area(f'{i+1}번 인물 성격 및 말투',key=f't{i}',height=80)
            if name.strip(): chars.append({'name':name.strip(),'traits':traits.strip()})
        ok=st.form_submit_button('작품 저장')
    if ok:
        if not title.strip() or not text.strip() or not chars: st.error('제목, 본문, 등장인물을 입력하세요.')
        else:
            works[title.strip()]={'title':title.strip(),'author':author.strip(),'text':text.strip(),'reference':ref.strip(),'characters':chars}; save(works); st.success('작품을 저장했습니다.')
    st.divider(); st.subheader('저장된 작품')
    for title,w in works.items():
        with st.expander(title):
            st.write('작가:',w.get('author','')); st.write('등장인물:',', '.join(x['name'] for x in w.get('characters',[])))
            if st.button('삭제',key='del'+title): del works[title]; save(works); st.rerun()

def student():
    works=load(); st.title('📖 문학 인물 인터뷰'); st.caption('AI의 답변을 그대로 믿지 말고 소설 본문과 비교해 보세요.')
    if not works: st.warning('교사가 먼저 작품을 등록해야 합니다.'); return
    if 'selected_work' not in st.session_state:
        title=st.selectbox('소설을 선택하세요',list(works)); names=[x['name'] for x in works[title]['characters']]; c=st.selectbox('등장인물을 선택하세요',names)
        if st.button('인터뷰 시작',type='primary'): start(title,c); st.rerun()
        return
    title=st.session_state.selected_work; c=st.session_state.character; w=works.get(title)
    cur=st.session_state.current_turn; dots=' '.join('●' if i<=cur else '○' for i in range(1,6)); st.markdown(f'### {title} · {c}'); st.markdown(f'**질문 {cur} / 5**  {dots}'); st.divider()
    for m in st.session_state.messages:
        with st.chat_message('user' if m['role']=='student' else 'assistant',avatar=None if m['role']=='student' else '📖'): st.write(m['content'])
    if st.session_state.finished:
        st.success('5번의 인터뷰가 끝났습니다. 이제 본문을 확인하세요.')
        name=st.text_input('이름 또는 닉네임'); guess=st.number_input('틀렸다고 생각하는 답변 번호',1,5,1); evidence=st.text_area('본문 근거',height=160)
        if st.button('내 답안 확인',type='primary'):
            judgment='정확함' if int(guess)==st.session_state.mistake_turn else '다시 확인 필요'; st.session_state.judgment=judgment
            (st.success if judgment=='정확함' else st.warning)(('정확합니다. 본문을 근거로 오답을 찾았습니다.' if judgment=='정확함' else '선택한 답변은 실제 의도된 오답이 아닙니다. 본문을 다시 확인해 보세요.'))
            st.write('**내가 적은 근거**'); st.write(evidence or '입력하지 않음')
            st.download_button('📥 인터뷰 결과 PNG 저장',result_png(title,c,name,st.session_state.messages,int(guess),evidence,judgment),'문학_인물_인터뷰_결과.png','image/png')
        if st.button('다시 인터뷰하기'): start(title,c); st.rerun()
        return
    q=st.chat_input(f'{c}에게 질문하세요. ({cur+1}번째 질문)')
    if q:
        st.session_state.messages.append({'role':'student','content':q}); turn=cur+1; st.session_state.current_turn=turn
        try:
            ans=make_mistake(w,c,q) if turn==st.session_state.mistake_turn else normal(w,c,q)
            st.session_state.mistake_used |= turn==st.session_state.mistake_turn; st.session_state.messages.append({'role':'character','content':ans}); st.session_state.finished=turn==5; st.rerun()
        except Exception as e:
            st.session_state.messages.pop(); st.session_state.current_turn=cur; st.error(str(e))

if 'page' not in st.session_state: st.session_state.page='student'
with st.sidebar:
    choice=st.radio('화면 선택',['학생 화면','교사 관리자'],index=0 if st.session_state.page=='student' else 1)
    st.session_state.page='student' if choice=='학생 화면' else 'admin'
if st.session_state.page=='admin': admin()
else: student()
