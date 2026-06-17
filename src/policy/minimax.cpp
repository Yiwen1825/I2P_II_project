#include <utility>
#include <algorithm>
#include "state.hpp"
#include "minimax.hpp"


/*============================================================
 * MiniMax — qsearch
 *
 * Quiescence search: called at depth 0 to avoid horizon effect.
 * Computes stand-pat score first; then searches only captures
 * until the position is quiet.
 *============================================================*/
int MiniMax::qsearch(
    State* state,
    int alpha,
    int beta,
    int ply,
    SearchContext& ctx,
    const MMParams& p
){
    ctx.nodes++;
    if(ply > ctx.seldepth) ctx.seldepth = ply;
    if(ctx.stop) return 0;

    /* Lazy move generation */
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    /* Terminal checks */
    if(state->game_state == WIN)  return P_MAX - ply;
    if(state->game_state == DRAW) return 0;

    /* Stand-pat: static eval without mobility (for speed) */
    int stand_pat = state->evaluate(p.use_kp_eval, false, nullptr);
    if(stand_pat >= beta) return stand_pat;
    if(stand_pat > alpha) alpha = stand_pat;

    /* Search captures only */
    auto oppn_board = state->board.board[1 - state->player];
    for(auto& action : state->legal_actions){
        if(oppn_board[action.second.first][action.second.second] == 0) continue;

        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();
        int raw = qsearch(next,
                          same ? alpha : -beta,
                          same ? beta  : -alpha,
                          ply + 1, ctx, p);
        int score = same ? raw : -raw;
        delete next;

        if(score > alpha) alpha = score;
        if(alpha >= beta) break;
    }

    return alpha;
}


/*============================================================
 * MiniMax — eval_ctx
 *
 * Negamax with Alpha-Beta + PVS. Caller manages memory.
 *============================================================*/
int MiniMax::eval_ctx(
    State *state,
    int depth,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const MMParams& p,
    int alpha,
    int beta
){
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }
    if(ctx.stop){
        return 0;
    }

    /* === Lazy move generation (sets game_state) === */
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    /* === Terminal / leaf checks === */

    // [ Hackathon TODO 3-1 ]
    if(state->game_state == WIN){
        return P_MAX - ply;
    }

    if(state->game_state == DRAW){
        return 0;
    }

    /* === Repetition check (game-specific) === */
    int rep_score;
    if(state->check_repetition(history, rep_score)){
        return rep_score;
    }
    history.push(state->hash());

    if(depth <= 0){
        /* Quiescence search instead of static eval */
        int score = qsearch(state, alpha, beta, ply, ctx, p);
        history.pop(state->hash());
        return score;
    }

    /* === Negamax loop with PVS === */
    int best_score = M_MAX;
    int move_index = 0;

    auto oppn_board = state->board.board[1 - state->player];
    auto ordered = state->legal_actions;
    std::sort(ordered.begin(), ordered.end(), [&](const Move& a, const Move& b){
        int va = (int)(unsigned char)oppn_board[a.second.first][a.second.second];
        int vb = (int)(unsigned char)oppn_board[b.second.first][b.second.second];
        return va > vb;
    });

    for(auto& action : ordered){
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();

        int score;
        if(!same && move_index > 0){
            /* PVS: null window search for non-first moves */
            int raw = eval_ctx(next, depth - 1, history, ply + 1, ctx, p,
                               -alpha - 1, -alpha);
            score = -raw;
            if(score > alpha && score < beta){
                /* Failed high: re-search with full window */
                raw = eval_ctx(next, depth - 1, history, ply + 1, ctx, p,
                               -beta, -alpha);
                score = -raw;
            }
        } else {
            /* First move or same-player: full window */
            int raw = eval_ctx(next, depth - 1, history, ply + 1, ctx, p,
                               same ? alpha : -beta,
                               same ? beta  : -alpha);
            score = same ? raw : -raw;
        }

        delete next;

        if(score > best_score) best_score = score;
        if(score > alpha) alpha = score;
        if(alpha >= beta) break; // alpha-beta prune

        move_index++;
    }

    history.pop(state->hash());
    return best_score;
}


/*============================================================
 * MiniMax — search
 *
 * Iterate legal moves, call eval_ctx, return SearchResult.
 *============================================================*/
SearchResult MiniMax::search(
    State *state,
    int depth,
    GameHistory& history,
    SearchContext& ctx
){
    ctx.reset();
    MMParams p = MMParams::from_map(ctx.params);
    SearchResult result;
    result.depth = depth;

    if(!state->legal_actions.size()){
        state->get_legal_actions();
    }

    int best_score = M_MAX - 10;
    int alpha = M_MAX;
    int beta  = P_MAX;
    int move_index = 0;
    int total_moves = (int)state->legal_actions.size();

    /* Sort root moves: captures first (MVV), then quiet moves */
    auto oppn_board_root = state->board.board[1 - state->player];
    auto ordered_root = state->legal_actions;
    std::sort(ordered_root.begin(), ordered_root.end(), [&](const Move& a, const Move& b){
        int va = (int)(unsigned char)oppn_board_root[a.second.first][a.second.second];
        int vb = (int)(unsigned char)oppn_board_root[b.second.first][b.second.second];
        return va > vb;
    });

    for(auto& action : ordered_root){
        // [ Hackathon TODO 4-1 ]
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();
        int score;
        if(!same && move_index > 0){
            /* PVS at root: null window for non-first moves */
            int raw = eval_ctx(next, depth - 1, history, 1, ctx, p, -alpha - 1, -alpha);
            score = -raw;
            if(score > alpha && score < beta){
                raw = eval_ctx(next, depth - 1, history, 1, ctx, p, -beta, -alpha);
                score = -raw;
            }
        } else {
            int raw = eval_ctx(next, depth - 1, history, 1, ctx, p,
                               same ? alpha : -beta, same ? beta : -alpha);
            score = same ? raw : -raw;
        }
        delete next;

        if(score > best_score){
            best_score = score;
            result.best_move = action;
            if(score > alpha) alpha = score;

            if(p.report_partial && ctx.on_root_update){
                ctx.on_root_update({result.best_move, best_score, depth, move_index + 1, total_moves});
            }
        }
        move_index++;
    }

    // [ Hackathon TODO 4-3 ]
    result.score = best_score;
    result.nodes = ctx.nodes;
    result.seldepth = ctx.seldepth;
    if(!state->legal_actions.empty())
        result.pv = {result.best_move};
    return result;
}


/*============================================================
 * MiniMax — default_params / param_defs
 *============================================================*/
ParamMap MiniMax::default_params(){
    return {
        {"UseKPEval", "true"},
        {"UseEvalMobility", "true"},
        {"ReportPartial", "true"},
    };
}

std::vector<ParamDef> MiniMax::param_defs(){
    return {
        {"UseKPEval", ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"ReportPartial", ParamDef::CHECK, "true"},
    };
}
