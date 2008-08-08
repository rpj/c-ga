/* gcc -g -lm -o ga ga.c
 * $Id: ga.c,v 1.5 2003/08/21 22:13:54 rjoseph Exp $ */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <math.h>
#include <time.h>
#include <string.h>

typedef unsigned char uchar;

int      ga_pop_sort_cmp(const void *, const void *);
uint     ga_get_random();
uchar    ga_sel_parent(uchar *, float *);
char   * bin2str(char *, char);
float    ga_fitness_all_ones(uchar);
void     ga_do_gen(uchar (*)(uchar *, float *), void (*)(uchar, uchar, uchar *),
         void (*)(uchar *), uchar *, float *, uchar *);
void     ga_crossover(uchar, uchar, uchar *);
void     ga_mutate(uchar *);
void     ga_eval_fitness(float (*)(uchar), uchar *, float *);
void     ga_gen_pop(uchar *);
void     ga_print_pop(uchar *);
void     help(char *);

// icky globals
unsigned long  ga_mutation_count = 0, ga_crossover_count = 0,
               ga_generation_count = 0, POP_SIZE = 200, PRINT_MOD = 11,
               GEN_COUNT = 10000, DUMP_DATA = 0;
double         PROB_CROSS = 0.7, PROB_MUTATE = 0.01, CUTOFF_REQ = 0.03;

#define OPT_STR   "ahqds:p:c:m:r:g:M:C:l:L:"

#ifdef DEBUG
# define dprint(fmt, args...) fprintf(stderr, ">> " fmt, ## args); 
#else
# define dprint(fmt, args...)
#endif

// start GA functions 
void ga_gen_pop(uchar *array) 
{
   char str;
   int i;
   
   srand(ga_get_random() * time(NULL));
   srand48(ga_get_random() * time(NULL));

   for (i = 0; i < POP_SIZE; i++) 
      array[i] = (uchar)((rand() >> (rand() % 3)) & 0xFF);

   qsort(array, POP_SIZE, sizeof(uchar), ga_pop_sort_cmp);
}

void ga_eval_fitness(float (*func)(uchar), uchar *pop, float *fitness) 
{
   int i;

   for (i = 0; i < POP_SIZE; i++)
      fitness[i] = (*func)(pop[i]);
}

float ga_fitness_all_ones(uchar indiv) { return (indiv / 256.0); }

int ga_pop_sort_cmp(const void *one, const void *two) 
{
   return (*((uchar *)one) - *((uchar *)two));
}

void ga_do_gen(uchar (*fselp)(uchar *, float *),
 void (*fcross)(uchar, uchar, uchar *), void (*fmutate)(uchar *),
 uchar *cur_pop, float *fit, uchar *new_pop)
{
   uchar *kids = (uchar *)(malloc(sizeof(uchar) * 2));
   int i;

   for (i = 0; i < POP_SIZE; i += 2) {
      (*fcross)((*fselp)(cur_pop, fit), (*fselp)(cur_pop, fit), kids);
      (*fmutate)(kids);
      
      new_pop[i]     = kids[0];
      new_pop[i+1]   = kids[1];
   }

   qsort(new_pop, POP_SIZE, sizeof(uchar), ga_pop_sort_cmp);
}

void ga_crossover(uchar p1, uchar p2, uchar *ret) 
{
   int locus, i, mask = 0;
   double cross = drand48();
   char str[256], str2[256];
   uchar k1 = p1, k2 = p2;

   if (cross < PROB_CROSS) {
      ga_crossover_count++;
      locus = (drand48() * 7) + 1;

      for (i = 0; i < locus + 1; i++)
         mask += pow(2, i);

      // these are the actual crossovers, don't know if this
      // is correct, but it does actually do stuff, so it helps
      k1 = (p1 & ~mask) | (p2 & mask);
      k2 = (p2 & ~mask) | (p1 & mask);
   }
   
   ret[0] = k1;
   ret[1] = k2;
}

void ga_mutate(uchar *kids)
{
   int locus, ndx;
   double mutate = drand48();
   uchar b4;
   
   if (mutate < PROB_MUTATE) {
      ga_mutation_count++;
      locus = drand48() * 8;
      ndx   = drand48() * 2;

      b4 = kids[ndx];
      kids[ndx] = kids[ndx] ^ (int)(pow(2.0, (double)locus));
   }
}

uchar ga_sel_parent(uchar *pop, float *fit) 
{
   double r1 = drand48(), sel_req;
   int sel;
   char parent;
   
   sel_req = (r1 + ((1.0 - r1) * 0.2));
   if (sel_req > fit[POP_SIZE - 1])
      sel_req = fit[POP_SIZE - 1];

   while (1) {
      sel = (int)(drand48() * POP_SIZE);

      if (fit[sel] >= sel_req)
         return pop[sel];
   }
}

// GA utility functions, not central to the GA
unsigned int ga_get_random() 
{
   int fd, ret;

   fd = open("/dev/urandom", O_RDONLY);
   read(fd, &ret, sizeof(ret));

   return (unsigned int)ret;
}

void ga_print_pop(uchar *pop)
{
   int i;

   printf("     ");
   for (i = 0; i < PRINT_MOD; i++)
      printf(" %8d", i + 1);
   printf("\n");
   
   printf("    +-----");
   for (i = 5; i < PRINT_MOD * 9; i++)
      printf("-");

   for (i = 0; i < POP_SIZE; i++) {
      if (!(i % PRINT_MOD))
         printf("\n%3d | ", i);
      
      printf("%8d ", pop[i]);
   }

   printf("\n");
}

// Regular util functions
char *bin2str(char *bin, char num) 
{
   int i;
   char mask;

   for (i = 0; i < 8; i++) {
      mask = pow(2, i);
      bin[(7 - i)] = ((num & mask) ? '1': '0');
   }

   bin[8] = '\0';
   return bin;      
}

void help(char *name) {
   printf(
   "Usage: %s [options]\n"
   "(C) 2003, Ryan Joseph <j@seph.us>\n"
   "Anything that is a percentage can be floating point.\n\n"
   "  -h    This help\n"
   "  -s    Population size (%d)\n"
   "  -p    Number of columns to print in pop tables (%d)\n"
   "  -r    Cutoff percentage for most-fit individuals (%0.2f)\n"
   "  -m    Mutation probability as a percentage (%0.2f)\n"
   "  -c    Crossover probability as a percentage (%0.2f)\n"
   "  -g    Number of generations (%d)\n"
   "  -a    Automated mode, using steps set below\n"
   "  -M    Mutation stepping, percentage\n"
   "  -C    Crossover stepping, percentage\n"
   "  -l    Mutation limit, percentage\n"
   "  -L    Crossover limit, percentage\n"
   "  -q    Don't print population tables\n\n",
   name, POP_SIZE, PRINT_MOD, (CUTOFF_REQ * 100.0),
   (PROB_MUTATE * 100.0), (PROB_CROSS * 100.0), GEN_COUNT);

   exit(0);
}

int main(int argc, char **argv) 
{
   uchar *pop, *gen, *opop;
   float *fit;
   int i = 0, tot_inds, no_pop_tables = 0, automated = 0;
   double mutate_step = 0.1, crossover_step = 0.1, ms, cs, lm, ls;
   char c;

   printf("Command line: ");
   for (i; i < argc; i++)
      printf("%s ", argv[i]);
   printf("\n");
   
   while ((c = getopt(argc, argv, OPT_STR)) != -1) {
      switch (c) {
      case 'h': help(argv[0]); break;
      case 's': POP_SIZE = atof(optarg); break;
      case 'p': PRINT_MOD = atoi(optarg); break;
      case 'r': CUTOFF_REQ = (atof(optarg) / 100.0); break;
      case 'm': PROB_MUTATE = (atof(optarg) / 100.0); break;
      case 'c': PROB_CROSS = (atof(optarg) / 100.0); break;
      case 'g': GEN_COUNT = atoi(optarg); break;
      case 'q': no_pop_tables = 1; break;
      case 'a': automated = 1; break;
      case 'M': mutate_step = (atof(optarg) / 100); break;
      case 'C': crossover_step = (atof(optarg) / 100); break;
      case 'l': lm = (atof(optarg) / 100); break;
      case 'L': ls = (atof(optarg) / 100); break;
      case 'd': DUMP_DATA = 1; break;
      default: break;
      }
   }
   
   if (!automated)
      lm = PROB_MUTATE + mutate_step, ls = PROB_CROSS + crossover_step;
  
   pop = (uchar *)(malloc(sizeof(uchar) * POP_SIZE));
   gen = (uchar *)(malloc(sizeof(uchar) * POP_SIZE));
   fit = (float *)(malloc(sizeof(float) * POP_SIZE));  

   for (ms = PROB_MUTATE, cs = PROB_CROSS; ms < lm || cs < ls;) {
      PROB_MUTATE = ms;
      PROB_CROSS = cs;

      if (automated) printf("\n----------\n");
      printf("Generations: %3d\tPopulation Size: %3d\tCutoff Percentage: %0.6f%%\n", 
            GEN_COUNT, POP_SIZE, (CUTOFF_REQ * 100.0));         
      printf("Crossover Probability: %3.2f%%\tMutation Probability: %3.6f%%\n",
            (PROB_CROSS * 100.0), (PROB_MUTATE * 100.0));

      if (automated)
         printf("Mutation step [%0.2f], limit [%0.2f]\t"
          "Crossover step [%0.2f], limit [%0.2f]\n",
          mutate_step * 100, lm * 100, crossover_step * 100, ls * 100);

      ga_gen_pop(pop);
      opop = pop;

      if (!no_pop_tables) {
         printf("\nFirst population:\n");
         ga_print_pop(pop); printf("\n");
      }
      
      ga_eval_fitness(ga_fitness_all_ones, pop, fit);
      do {
         if (DUMP_DATA) {
            for (i = 0; i < POP_SIZE; i++)
               fprintf(stderr, "%d\t", pop[i]);

            fprintf(stderr, "\n");
         }
         
         dprint("Generation: %12d\tMutations: %12d\t Crossovers: %12d\n",
          ga_generation_count, (ga_mutation_count * 2), (ga_crossover_count * 2));
         ga_do_gen(  ga_sel_parent, 
                     ga_crossover,
                     ga_mutate,
                     pop, fit, gen
                     );
         
         pop = gen;
         ga_eval_fitness(ga_fitness_all_ones, pop, fit);
         ga_generation_count++;
      } while (ga_generation_count < GEN_COUNT &&
       pop[(int)(((float)(POP_SIZE * CUTOFF_REQ)) - 1)] != 255);

      if (!no_pop_tables) {
         printf("Final population:\n");
         ga_print_pop(pop);
         printf("\n");
      }

      tot_inds = ga_generation_count * POP_SIZE;
      printf("\nDone with %d generations (%d individuals)"
            "\n%d crossovers and %d mutations\n",
            ga_generation_count, tot_inds, ga_crossover_count, ga_mutation_count);

      printf("Used %0.1f%% of the allotted generations.\n\n",
       (((double)(ga_generation_count) / (double)GEN_COUNT) * 100.0));
      printf("Real crossover percentage (set as %0.2f%%): %0.3f%%\n",
       (PROB_CROSS * 100.0), ((double)(ga_crossover_count * 2) /
       (double)tot_inds) * 100.0);
      printf("Real mutation percentage (set as %0.3f%%): %0.4f%%\n",
       (PROB_MUTATE * 100.0),((double)(ga_mutation_count * 2) /
       (double)tot_inds) * 100.0);

      for (i = 0; i < POP_SIZE && pop[i] < 255; i++) {}
      printf("Number of 'most-fit' individuals: %d (%0.5f%%)\n",
       (POP_SIZE - i), 100.0 - ((((double)(i + 1)) / POP_SIZE) * 100.0));      

      if (ms < lm) ms += mutate_step;
      if (cs < ls) cs += crossover_step;
   }

   free(opop);
   free(pop);
   free(fit);

   return 0;
}
