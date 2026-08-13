/* gate-tool ONLY — test harness for lexer.s, not part of compiler layers */
#include <stdio.h>
#include <string.h>
#include <stdint.h>

extern long fl_lex(const char *src, unsigned long len, uint64_t *out, unsigned long cap)
    __asm__("_fl_lex");

static int fails = 0;
static void chk(const char *name, int ok) {
    printf("%s %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) fails++;
}

int main(void) {
    uint64_t buf[256];

    /* 1: normal program */
    const char *s1 = "界 64 64 16 42\n種 0 7 # seed\n撃 3 4 1065353216\n歩 100\n縛 2 0 1\n束 3 2 0 1\n寫 5 1 9\n";
    long n = fl_lex(s1, strlen(s1), buf, 256);
    uint64_t exp[] = {9,64,64,16,42, 1,0,7, 2,3,4,1065353216, 3,100, 4,2,0,1, 5,3,2,0,1, 6,5,1,9};
    int ok = (n == 28);
    if (ok) {
        for (int i = 0; i < 27; i++) {
            uint64_t k = buf[2*i], v = buf[2*i+1];
            uint64_t want = exp[i];
            uint64_t got = (want < 10 && (i==0||buf[2*i] < 10)) ? 0 : 0; (void)got;
            /* flatten: glyph tokens value=0, ints kind=7 */
            if (k == 7) { if (v != want) ok = 0; }
            else { if (k != want || v != 0) ok = 0; }
        }
        if (buf[2*27] != 0 || buf[2*27+1] != 0) ok = 0; /* EOF record */
    }
    chk("normal", ok);

    /* 2: bad char line 3 */
    const char *s2 = "歩 1\n# c\nZ\n";
    n = fl_lex(s2, strlen(s2), buf, 256);
    chk("badchar", n == -((3l<<8)|1));

    /* 3: overflow line 2 */
    const char *s3 = "歩 1\n18446744073709551616\n";
    n = fl_lex(s3, strlen(s3), buf, 256);
    chk("overflow", n == -((2l<<8)|2));

    /* 4: buffull line 1 */
    const char *s4 = "歩 1 2 3";
    n = fl_lex(s4, strlen(s4), buf, 2);
    chk("buffull", n == -((1l<<8)|3));

    /* 5: u64 max ok + empty-ish comment-only */
    const char *s5 = "18446744073709551615 # max\n";
    n = fl_lex(s5, strlen(s5), buf, 256);
    chk("u64max", n == 2 && buf[0] == 7 && buf[1] == UINT64_MAX && buf[2] == 0);

    /* 6: unknown 3-byte glyph -> badchar */
    const char *s6 = "愛";
    n = fl_lex(s6, strlen(s6), buf, 256);
    chk("badglyph", n == -((1l<<8)|1));

    /* 7: truncated glyph at EOF -> badchar */
    n = fl_lex("\xE7\xA8", 2, buf, 256);
    chk("truncglyph", n == -((1l<<8)|1));

    /* 8: empty input -> just EOF */
    n = fl_lex("", 0, buf, 256);
    chk("empty", n == 1 && buf[0] == 0 && buf[1] == 0);

    printf(fails ? "GATE RED (%d)\n" : "GATE GREEN\n", fails);
    return fails != 0;
}
