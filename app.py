        st.subheader(f"{work.get('title','')} · {character}")
        st.caption(f"{work.get('author','')} 작품")
    with top_right:
        st.markdown(
            f'<div class="progress-box">질문 {min(turn_count + 1, MAX_TURNS)} / {MAX_TURNS}</div>',
            unsafe_allow_html=True,
        )

    dots = " ".join("●" if i <= turn_count else "○" for i in range(1, MAX_TURNS + 1))
    st.markdown(f"**진행:** {dots}")

    for idx, item in enumerate(st.session_state.history, start=1):
        with st.chat_message("user"):
            st.markdown(f"**Q{idx}.** {item['question']}")
        with st.chat_message("assistant"):
            st.markdown(item["answer"])

    if turn_count < MAX_TURNS and not st.session_state.finished:
        with st.form("question_form", clear_on_submit=True):
            question = st.text_input(
                "질문",
                max_chars=300,
                placeholder=f"{character}에게 궁금한 점을 한 가지 질문해 보세요.",
                disabled=st.session_state.busy,
            )
            submitted = st.form_submit_button(
                "질문하기",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.busy,
            )

        if submitted:
            q = (question or "").strip()
            if not q:
                st.warning("질문을 입력해 주세요.")
            else:
                st.session_state.busy = True
                current_turn = turn_count + 1
                make_mistake = current_turn == st.session_state.mistake_turn
                try:
                    with st.spinner(f"{character}가 답변을 생각하고 있어요..."):
                        answer, used_search, search_error = generate_answer_with_validation(
                            work,
                            character,
                            q,
                            st.session_state.history,
                            make_mistake,
                        )
                    if not answer:
                        raise RuntimeError("빈 답변이 생성되었습니다.")

                    st.session_state.history.append(
                        {
                            "question": q,
                            "answer": answer,
                            "character": character,
                        }
                    )

                    if search_error and not used_search:
                        st.session_state.last_search_notice = (
                            "교사 제공 소설 파일이 없어 웹 검색을 시도했지만, 현재 API 설정에서는 검색을 사용할 수 없어 "
                            "Gemini 자체 지식으로 답했습니다."
                        )

                    if len(st.session_state.history) >= MAX_TURNS:
                        st.session_state.finished = True
                    st.session_state.busy = False
                    st.rerun()
                except Exception as e:
                    st.session_state.busy = False
                    st.error(f"답변을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요. ({e})")

    if st.session_state.last_search_notice:
        st.caption("※ " + st.session_state.last_search_notice)

    if st.session_state.finished:
        st.divider()
        st.subheader("🔎 이제 AI의 오류를 찾아보세요")
        st.write("5개의 답변 중 작품 내용과 어긋나는 답변은 **딱 하나**입니다.")

        with st.form("result_form"):
            picked = st.selectbox("틀린 답변이라고 생각하는 번호", list(range(1, MAX_TURNS + 1)))
            evidence = st.text_area(
                "작품 속 근거",
                placeholder="왜 그 답변이 틀렸는지 작품의 사건, 행동, 대사 등을 근거로 적어 보세요.",
                height=130,
            )
            check = st.form_submit_button("확인하기", type="primary", use_container_width=True)

        if check:
            if not evidence.strip():
                st.warning("작품 속 근거를 적어 주세요.")
            elif picked == st.session_state.mistake_turn:
                st.success("오답의 위치를 정확히 찾았습니다! 이제 작품 근거가 적절한지 확인해 볼게요.")
                try:
                    with st.spinner("작품 근거를 확인하고 있어요..."):
                        feedback = evaluate_evidence(
                            work,
                            character,
                            st.session_state.history[picked - 1],
                            evidence.strip(),
                        )
                    st.info(feedback)
                except Exception:
                    st.info("오답 번호는 맞았습니다. 작품 근거는 수업에서 친구들과 함께 다시 확인해 보세요.")

                png = make_result_png(
                    work,
                    character,
                    st.session_state.history,
                    picked,
                    evidence.strip(),
                )
                st.download_button(
                    "결과를 PNG로 저장",
                    data=png,
                    file_name=f"{work.get('title','작품')}_인물인터뷰_결과.png",
                    mime="image/png",
                    use_container_width=True,
                )
            else:
                st.warning("다시 생각해 보세요. 작품 속 사건의 순서, 인물의 관계, 행동의 이유를 확인해 보세요.")

        if st.button("🔄 다시 인터뷰하기", use_container_width=True):
            reset_interview()
            st.rerun()
