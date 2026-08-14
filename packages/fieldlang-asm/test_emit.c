/* test_emit.c — GATE-TOOL ONLY (not part of compiler; C permitted per CONTRACT
 * "test-only C or shell OK, marked gate-tool"). Hand-builds token records,
 * calls _fl_emit, compares bytes, dumps outputs for xxd inspection. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

typedef struct { uint64_t kind, value; } Tok;
extern long _fl_emit(const Tok*, long, uint8_t*, long, const void*)
    __asm__("_fl_emit");

/* defaults struct per CONTRACT: {u32 w,h; u32 c,dt,damp,dx; u64 seed; u32 range; u32 n_slots} */
static const struct { uint32_t w,h,c,dt,damp,dx; uint64_t seed; uint32_t range,n_slots; }
defaults = {64,64,0x3F800000,0x3DCCCCCD,0x3F7FBE77,0x3F800000,42,0x3F800000,16};

static int fails = 0;
static void check(const char *name, long got, long want_len,
                  const uint8_t *buf, const uint8_t *exp) {
    if (got != want_len || (want_len > 0 && memcmp(buf, exp, want_len))) {
        printf("FAIL %s: ret=%ld want=%ld\n", name, got, want_len);
        fails++;
    } else printf("PASS %s (ret=%ld)\n", name, got);
    if (got > 0) { char fn[64]; snprintf(fn, sizeof fn, "gate_%s.bin", name);
        FILE *f = fopen(fn, "wb");
        if (f) { fwrite(buf, 1, (size_t)got, f); fclose(f); } }
}

static void hdr_default(uint8_t *e) {
    uint32_t v[12] = {0x4A444C46,1,64,64,0x3F800000,0x3DCCCCCD,0x3F7FBE77,
                      0x3F800000,42,0,0x3F800000,16};
    memcpy(e, v, 48);
}

int main(void) {
    uint8_t buf[4096], exp[4096];

    /* T1: EOF only → 48B default header */
    { Tok t[] = {{0,0}};
      hdr_default(exp);
      check("t1_defaults", _fl_emit(t,1,buf,sizeof buf,&defaults), 48, buf, exp); }

    /* T2: 界 128 32 8 7 → header override */
    { Tok t[] = {{9,0},{7,128},{7,32},{7,8},{7,7},{0,0}};
      hdr_default(exp);
      ((uint32_t*)exp)[2]=128; ((uint32_t*)exp)[3]=32;
      memcpy(exp+32,(uint64_t[]){7},8); ((uint32_t*)exp)[11]=8;
      check("t2_kai", _fl_emit(t,6,buf,sizeof buf,&defaults), 48, buf, exp); }

    /* T3: one of each fixed op: 種 3 99 · 撃 1 2 0x3F800000 · 歩 10 · 縛 5 3 4 */
    { Tok t[] = {{1,0},{7,3},{7,99},
                 {2,0},{7,1},{7,2},{7,0x3F800000},
                 {3,0},{7,10},
                 {4,0},{7,5},{7,3},{7,4},{0,0}};
      hdr_default(exp); uint8_t *p = exp+48;
      *p++=1; memcpy(p,(uint32_t[]){3},4);p+=4; memcpy(p,(uint64_t[]){99},8);p+=8;
      *p++=2; memcpy(p,(uint32_t[]){1,2,0x3F800000},12);p+=12;
      *p++=3; memcpy(p,(uint32_t[]){10},4);p+=4;
      *p++=4; memcpy(p,(uint32_t[]){5,3,4},12);p+=12;
      check("t3_ops", _fl_emit(t,14,buf,sizeof buf,&defaults), p-exp, buf, exp); }

    /* T4: variable length: 界 2 2 16 42 · 束 9 3 1 2 3 · 寫 2 4 100 200 300 400
     *     (寫 n MUST equal w*h — hence the 2x2 界) */
    { Tok t[] = {{9,0},{7,2},{7,2},{7,16},{7,42},
                 {5,0},{7,9},{7,3},{7,1},{7,2},{7,3},
                 {6,0},{7,2},{7,4},{7,100},{7,200},{7,300},{7,400},{0,0}};
      hdr_default(exp); ((uint32_t*)exp)[2]=2; ((uint32_t*)exp)[3]=2;
      uint8_t *p = exp+48;
      *p++=5; memcpy(p,(uint32_t[]){9,3,1,2,3},20);p+=20;
      *p++=6; memcpy(p,(uint32_t[]){2,4,100,200,300,400},24);p+=24;
      check("t4_varlen", _fl_emit(t,19,buf,sizeof buf,&defaults), p-exp, buf, exp); }

    /* T4b: 寫 shape mismatch (n != w*h) → -5 ; matching n → accepted */
    { Tok bad[] = {{9,0},{7,2},{7,2},{7,16},{7,42},
                   {6,0},{7,0},{7,3},{7,1},{7,2},{7,3},{0,0}};
      long r=_fl_emit(bad,12,buf,sizeof buf,&defaults);
      printf("%s t4b_shape_mismatch (%ld)\n", r==-5?"PASS":"FAIL", r); if(r!=-5)fails++;
      Tok ok[] = {{9,0},{7,2},{7,2},{7,16},{7,42},
                  {6,0},{7,0},{7,4},{7,1},{7,2},{7,3},{7,4},{0,0}};
      r=_fl_emit(ok,13,buf,sizeof buf,&defaults);
      printf("%s t4b_shape_ok (%ld)\n", r==48+1+24?"PASS":"FAIL", r); if(r!=48+1+24)fails++; }

    /* T5: syntax errors → -1 */
    { Tok a[] = {{7,5},{0,0}};                 /* bare INT at op position */
      Tok b[] = {{3,0},{0,0}};                 /* 歩 missing arg (EOF as arg) */
      Tok c[] = {{3,0},{7,1},{9,0},{7,1},{7,1},{7,1},{7,1},{0,0}}; /* 界 not first */
      Tok d[] = {{3,0},{7,1}};                 /* no EOF record */
      long r;
      r=_fl_emit(a,2,buf,sizeof buf,&defaults); printf("%s t5a bare-int (%ld)\n", r==-1?"PASS":"FAIL", r); if(r!=-1)fails++;
      r=_fl_emit(b,2,buf,sizeof buf,&defaults); printf("%s t5b missing-arg (%ld)\n", r==-1?"PASS":"FAIL", r); if(r!=-1)fails++;
      r=_fl_emit(c,8,buf,sizeof buf,&defaults); printf("%s t5c kai-late (%ld)\n", r==-1?"PASS":"FAIL", r); if(r!=-1)fails++;
      r=_fl_emit(d,2,buf,sizeof buf,&defaults); printf("%s t5d no-eof (%ld)\n", r==-1?"PASS":"FAIL", r); if(r!=-1)fails++; }

    /* T6: arg range → -4 (x > u32 max) */
    { Tok t[] = {{3,0},{7,0x100000000ULL},{0,0}};
      long r=_fl_emit(t,3,buf,sizeof buf,&defaults);
      printf("%s t6_range (%ld)\n", r==-4?"PASS":"FAIL", r); if(r!=-4)fails++; }

    /* T7: buffull → -2 (cap 47 < header; cap 50 < header+op) */
    { Tok t[] = {{3,0},{7,1},{0,0}};
      long r=_fl_emit(t,3,buf,47,&defaults);
      printf("%s t7a buffull-hdr (%ld)\n", r==-2?"PASS":"FAIL", r); if(r!=-2)fails++;
      r=_fl_emit(t,3,buf,50,&defaults);
      printf("%s t7b buffull-op (%ld)\n", r==-2?"PASS":"FAIL", r); if(r!=-2)fails++; }

    printf(fails ? "GATE RED (%d fails)\n" : "GATE GREEN\n", fails);
    return fails ? 1 : 0;
}
